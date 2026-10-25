"""Proper joint mixture scoring, masked marginals and permutation-aware vision targets."""

from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F

from .config import Config, ModelConfig
from .schema import EvidenceBatch, JointBelief, ModelOutput, TrainingTargets


def scales(cfg: ModelConfig, device) -> Tensor:
    return torch.tensor([cfg.translation_scale] * 3 + [cfg.rotation_scale] * 3, device=device)


def mixture_log_prob(logits: Tensor, component_log_prob: Tensor) -> Tensor:
    return torch.logsumexp(logits.log_softmax(-1) + component_log_prob, -1)


def lowrank_marginal_log_prob(
    mean: Tensor, diagonal: Tensor, factor: Tensor, value: Tensor, valid: Tensor
) -> Tensor:
    """Exact marginal over labeled dimensions, retaining covariance across all nodes.

    Returns [K]. An empty target contributes zero (no spurious log-weight penalty).
    Inputs are one scene's [K,D], [K,D], [K,D,R], [D], [D].
    """
    selected = valid.bool()
    if not selected.any():
        return mean.sum(-1) * 0
    distribution = torch.distributions.LowRankMultivariateNormal(
        mean[:, selected].float(), factor[:, selected].float(), diagonal[:, selected].float()
    )
    return distribution.log_prob(value[selected].float())


@torch.no_grad()
def match_entities(
    output: ModelOutput, evidence: EvidenceBatch, targets: TrainingTargets, cfg: ModelConfig
) -> list[tuple[Tensor, Tensor]]:
    """Whole-prefix mask assignment; never changes online slot state or input tokens."""
    b, t, s1, _, p = output.slot_masks.shape
    grid = int(math.sqrt(p))
    predicted = output.slot_masks[:, :, : cfg.slots, 0].reshape(b, t, cfg.slots, grid, grid)
    valid_pixels = (
        evidence.pixel_valid[:, :, 0]
        & evidence.frame_valid[..., None, None]
        & evidence.camera_valid[:, :, 0, None, None]
    )
    weights = F.adaptive_avg_pool2d(
        valid_pixels.float().flatten(0, 1)[:, None], (grid, grid)
    ).reshape(b, t, grid, grid)
    masked_masks = targets.instance_masks.float() * valid_pixels[:, :, None]
    actual = F.adaptive_avg_pool2d(masked_masks.flatten(0, 2)[:, None], (grid, grid)).reshape(
        b, t, -1, grid, grid
    )
    matches = []
    for i in range(b):
        obj = targets.entity_valid[i].nonzero().flatten()
        if obj.numel() > cfg.slots:
            raise ValueError("target objects exceed slot capacity; do not silently truncate")
        pred = predicted[i].transpose(0, 1).flatten(1)
        truth = actual[i, :, obj].transpose(0, 1).flatten(1)
        weighted_pred = (predicted[i] * weights[i, :, None]).transpose(0, 1).flatten(1)
        if obj.numel() == 0:
            matches.append((obj, obj))
            continue
        intersection = pred @ truth.T
        cost = 1 - (2 * intersection + 1) / (
            weighted_pred.sum(-1)[:, None] + truth.sum(-1)[None] + 1
        )
        slots_idx, objects_idx = linear_sum_assignment(cost.cpu().numpy())
        matches.append((torch.as_tensor(slots_idx, device=pred.device), obj[objects_idx]))
    return matches


def aligned_pose_targets(targets: TrainingTargets, matches, cfg: ModelConfig):
    b = targets.poses.shape[0]
    values = targets.poses.new_zeros(b, cfg.slots, 6)
    valid = torch.zeros_like(values, dtype=torch.bool)
    for i, (slots, objects) in enumerate(matches):
        values[i, slots] = targets.poses[i, objects]
        valid[i, slots] = targets.pose_valid[i, objects]
    return values / scales(cfg, values.device), valid


def state_nll(
    belief: JointBelief, targets: TrainingTargets, matches, cfg: ModelConfig
) -> tuple[Tensor, Tensor]:
    values, valid = aligned_pose_targets(targets, matches, cfg)
    scene_losses, counts = [], []
    for i, (slots, objects) in enumerate(matches):
        count = valid[i].sum()
        if not count:
            scene_losses.append(belief.mean[i].sum() * 0)
            counts.append(count)
            continue
        alternatives = [values[i]]
        if targets.equivalent_poses is not None:
            if targets.equivalent_valid is None:
                raise ValueError("equivalent poses require an explicit validity mask")
            alternatives = []
            complete_representatives = []
            for j in targets.equivalent_valid[i].nonzero().flatten():
                alt = values[i].clone()
                alt[slots] = targets.equivalent_poses[i, j, objects] / scales(cfg, values.device)
                if any(torch.allclose(alt, prior) for prior in complete_representatives):
                    raise ValueError("duplicate symmetry alternative would inflate probability")
                complete_representatives.append(alt)
                # Distinct full states can have the same marginal on observed coordinates.
                if not any(
                    torch.allclose(alt[valid[i]], prior[valid[i]]) for prior in alternatives
                ):
                    alternatives.append(alt)
            if not alternatives:
                raise ValueError("empty equivalence class")
        log_prob = torch.stack(
            [
                lowrank_marginal_log_prob(
                    belief.mean[i].flatten(-2),
                    belief.diagonal[i],
                    belief.factor[i],
                    alt.flatten(),
                    valid[i].flatten(),
                )
                for alt in alternatives
            ]
        )
        # Sum over explicitly disjoint symmetry representatives in this local chart.
        component = torch.logsumexp(log_prob, 0)
        scene_losses.append(-mixture_log_prob(belief.logits[i], component) / count.clamp_min(1))
        counts.append(count)
    return torch.stack(scene_losses).mean(), torch.stack(counts).sum()


def change_loss(
    belief: JointBelief, edge_index: Tensor, targets: TrainingTargets, matches
) -> tuple[Tensor, int]:
    # Conditional per-edge proper scores; not a claim of independent physical events.
    terms = []
    for b, (slots, objects) in enumerate(matches):
        mapping = dict(zip(slots.tolist(), objects.tolist()))
        for e, (left, right) in enumerate(edge_index.T.tolist()):
            if left not in mapping or right not in mapping:
                continue
            i, j = mapping[left], mapping[right]
            if targets.change_valid[b, i, j]:
                label = targets.change_times[b, i, j]
                if label < 0 or label >= belief.change_logits.shape[-1]:
                    raise ValueError("change time class out of range")
                lp = belief.change_logits[b, :, e].log_softmax(-1)[:, label]
                terms.append(-mixture_log_prob(belief.logits[b], lp))
    return (torch.stack(terms).mean(), len(terms)) if terms else (belief.logits.sum() * 0, 0)


def normal_log_prob(mean: Tensor, std: Tensor, value: Tensor) -> Tensor:
    return -0.5 * ((value - mean) / std).square() - std.log() - 0.5 * math.log(2 * math.pi)


def relation_attribute_loss(
    belief: JointBelief, edge_index: Tensor, targets: TrainingTargets, matches
) -> tuple[Tensor, int]:
    terms = []
    for b, (slots, objects) in enumerate(matches):
        mapping = dict(zip(slots.tolist(), objects.tolist()))
        for e, (left, right) in enumerate(edge_index.T.tolist()):
            if left not in mapping or right not in mapping:
                continue
            i, j = mapping[left], mapping[right]
            if (
                targets.relation_relevance is not None
                and targets.relation_relevance_valid is not None
                and targets.relation_relevance_valid[b, i, j]
            ):
                score = belief.relevance_logits[b, :, e]
                label = targets.relation_relevance[b, i, j].to(score.dtype).expand_as(score)
                component = -F.binary_cross_entropy_with_logits(score, label, reduction="none")
                terms.append(-mixture_log_prob(belief.logits[b], component))
            if (
                targets.contact_states is not None
                and targets.contact_valid is not None
                and targets.contact_valid[b, i, j]
            ):
                label = targets.contact_states[b, i, j]
                if label < 0 or label > 2:
                    raise ValueError("contact target must be unknown/no/yes")
                component = belief.contact_logits[b, :, e].log_softmax(-1)[:, label]
                terms.append(-mixture_log_prob(belief.logits[b], component))
    return (torch.stack(terms).mean(), len(terms)) if terms else (belief.logits.sum() * 0, 0)


def prediction_nll(
    pred: dict | None,
    belief: JointBelief,
    targets: TrainingTargets,
    appearance_target: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if pred is None:
        return belief.logits.sum() * 0, belief.logits.new_zeros(())
    response = normal_log_prob(
        pred["response_mean"], pred["response_std"], targets.future_response[:, None]
    )
    valid = targets.future_response_valid
    components = (response * valid[:, None]).sum(-1)
    counts = valid.sum(-1)
    if appearance_target is not None:
        visual_ok = targets.future_image_valid.any(-1)
        appearance = normal_log_prob(
            pred["appearance_mean"], pred["appearance_std"], appearance_target[:, None]
        ).sum(-1)
        components = components + appearance * visual_ok[:, None]
        counts = counts + visual_ok * appearance_target.shape[-1]
    success = -F.binary_cross_entropy_with_logits(
        pred["success_logits"],
        targets.future_success[:, None]
        .to(pred["success_logits"].dtype)
        .expand_as(pred["success_logits"]),
        reduction="none",
    )
    components = components + success * targets.future_success_valid[:, None]
    counts = counts + targets.future_success_valid
    loss = -mixture_log_prob(pred["mixture_logits"], components) / counts.clamp_min(1)
    return loss.mean(), counts.sum()


def vision_loss(
    output: ModelOutput,
    evidence: EvidenceBatch,
    targets: TrainingTargets,
    matches,
    cfg: ModelConfig,
) -> Tensor:
    b, t, _, _, p = output.slot_masks.shape
    grid = int(math.sqrt(p))
    valid_pixels = (
        evidence.pixel_valid[:, :, 0]
        & evidence.frame_valid[..., None, None]
        & evidence.camera_valid[:, :, 0, None, None]
    )
    pixel_weight = F.adaptive_avg_pool2d(
        valid_pixels.float().flatten(0, 1)[:, None], (grid, grid)
    ).reshape(b, t, p)
    masked_masks = targets.instance_masks.float() * valid_pixels[:, :, None]
    truth = F.adaptive_avg_pool2d(masked_masks.flatten(0, 2)[:, None], (grid, grid)).reshape(
        b, t, -1, p
    )
    truth = truth / pixel_weight[:, :, None].clamp_min(1e-8)
    pred = output.slot_masks[:, :, :, 0].clamp_min(1e-8)
    desired = torch.zeros_like(pred)
    presence = torch.zeros_like(output.presence_logits)
    for i, (slots, objects) in enumerate(matches):
        desired[i, :, slots] = truth[i, :, objects]
        presence[i, :, slots] = targets.visibility[i, :, objects].float()
    desired[:, :, -1] = (1 - desired[:, :, :-1].sum(2)).clamp_min(0)
    # Instance masks describe a visible partition. Reject overlapping labels.
    if (desired.sum(2) > 1.001).any():
        raise ValueError("instance supervision must use non-overlapping visible masks")
    segmentation = (
        -(desired * pred.log()).sum(2) * pixel_weight
    ).sum() / pixel_weight.sum().clamp_min(1)
    fv = evidence.frame_valid[..., None].expand_as(presence)
    presence_loss = (
        F.binary_cross_entropy_with_logits(output.presence_logits, presence, reduction="none") * fv
    ).sum() / fv.sum().clamp_min(1)
    valid_rgb = (
        evidence.pixel_valid
        & evidence.camera_valid[..., None, None]
        & evidence.frame_valid[..., None, None, None]
    ).unsqueeze(3)
    reconstruction = ((output.reconstruction - evidence.images).square() * valid_rgb).sum()
    reconstruction = reconstruction / (valid_rgb.sum() * 3).clamp_min(1)
    association_terms = []
    for i, (slots, objects) in enumerate(matches):
        for time in range(1, t):
            if not evidence.frame_valid[i, time] or not evidence.frame_valid[i, time - 1]:
                continue
            visible = targets.visibility[i, time, objects].bool()
            if visible.any():
                previous_visible = targets.visibility[i, time - 1, objects].bool()
                labels = torch.where(previous_visible, slots, cfg.slots)
                association_terms.append(
                    F.cross_entropy(
                        output.association_logits[i, time, slots[visible]], labels[visible]
                    )
                )
    association = torch.stack(association_terms).mean() if association_terms else segmentation * 0
    return segmentation + presence_loss + reconstruction + association


def geometry_loss(
    belief: JointBelief,
    evidence: EvidenceBatch,
    targets: TrainingTargets,
    matches,
    cfg: ModelConfig,
) -> tuple[Tensor, int]:
    if targets.image_points is None or targets.image_points_valid is None:
        return belief.logits.sum() * 0, 0
    terms = []
    scale = scales(cfg, belief.mean.device)
    for b, (slots, objects) in enumerate(matches):
        latest = evidence.frame_valid[b].nonzero()[-1, 0]
        if not evidence.metric_valid[b] or not evidence.camera_valid[b, latest, 0]:
            continue
        calibration = evidence.calibration[b, latest, 0]
        intrinsic = calibration[:9].reshape(3, 3)
        camera_to_base = calibration[9:].reshape(4, 4)
        base_to_camera = torch.linalg.inv(camera_to_base)
        ok = targets.image_points_valid[b, objects]
        if not ok.any():
            continue
        points = belief.mean[b, :, slots[ok], :3] * scale[:3]
        cam = points @ base_to_camera[:3, :3].T + base_to_camera[:3, 3]
        image = cam @ intrinsic.T
        projected = image[..., :2] / image[..., 2:].clamp_min(1e-5)
        # Auxiliary reprojection score in normalized image units, mode mixture preserved.
        error = (
            ((projected - targets.image_points[b, objects[ok]]) / cfg.image_size)
            .square()
            .sum((-1, -2))
        )
        error = error + (cam[..., 2] <= 0).float().sum(-1) * 100
        terms.append(-mixture_log_prob(belief.logits[b], -error) / ok.sum())
    return (torch.stack(terms).mean(), len(terms)) if terms else (belief.logits.sum() * 0, 0)


class ResearchLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        output: ModelOutput,
        evidence: EvidenceBatch,
        targets: TrainingTargets,
        appearance_target: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        cfg, weights = self.cfg.model, self.cfg.train
        matches = match_entities(output, evidence, targets, cfg)
        final_state, state_count = state_nll(output.final, targets, matches, cfg)
        first_state, _ = state_nll(output.first, targets, matches, cfg)
        final_change, change_count = change_loss(output.final, output.edge_index, targets, matches)
        first_change, _ = change_loss(output.first, output.edge_index, targets, matches)
        attributes, attribute_count = relation_attribute_loss(
            output.final, output.edge_index, targets, matches
        )
        first_attributes, _ = relation_attribute_loss(
            output.first, output.edge_index, targets, matches
        )
        prediction, prediction_count = prediction_nll(
            output.prediction, output.final, targets, appearance_target
        )
        first_prediction, _ = prediction_nll(
            output.first_prediction, output.first, targets, appearance_target
        )
        vision = vision_loss(output, evidence, targets, matches, cfg)
        geometry, geometry_count = geometry_loss(output.final, evidence, targets, matches, cfg)
        first_loss = first_state + first_change + first_attributes + first_prediction
        total = (
            weights.state_weight * (final_state + final_change + attributes)
            + weights.prediction_weight * prediction
            + weights.vision_weight * vision
            + weights.geometry_weight * geometry
            + weights.round1_weight * first_loss
        )
        terms = {
            "loss": total,
            "state_nll": final_state,
            "change_nll": final_change,
            "prediction_nll": prediction,
            "vision": vision,
            "geometry": geometry,
            "round1": first_loss,
            "state_dimensions": state_count,
            "change_labels": change_count,
            "prediction_dimensions": prediction_count,
            "geometry_labels": geometry_count,
        }
        terms.update(
            {"relation_attributes": attributes, "relation_attribute_labels": attribute_count}
        )
        return total, {
            name: float(value.detach()) if isinstance(value, Tensor) else float(value)
            for name, value in terms.items()
        }
