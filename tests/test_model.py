from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from rch_pfn.config import load_config
from rch_pfn.losses import ResearchLoss, match_entities
from rch_pfn.model import RCHPFN
from rch_pfn.schema import EvidenceBatch, TrainingTargets

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(SMOKE_CONFIG)


def _evidence_and_targets(cfg) -> tuple[EvidenceBatch, TrainingTargets]:
    model_cfg = cfg.model
    b, t, c, h = 1, 2, 1, model_cfg.image_size
    rng = torch.Generator().manual_seed(20261025)
    images = torch.rand((b, t, c, 3, h, h), generator=rng)
    frame_valid = torch.ones((b, t), dtype=torch.bool)
    camera_valid = torch.ones((b, t, c), dtype=torch.bool)
    pixel_valid = torch.ones((b, t, c, h, h), dtype=torch.bool)
    anchors = torch.eye(4).expand(b, t, 2, 4, 4).clone()
    evidence = EvidenceBatch(
        images=images,
        pixel_valid=pixel_valid,
        frame_valid=frame_valid,
        camera_valid=camera_valid,
        timestamps=torch.tensor([[0.0, 1.0]]),
        arrival_times=torch.tensor([[0.0, 1.0]]),
        query_time=torch.tensor([1.0]),
        actions=torch.rand((b, t, model_cfg.action_dim), generator=rng),
        robots=torch.rand((b, t, model_cfg.robot_dim), generator=rng),
        calibration=torch.zeros((b, t, c, 25)),
        task=torch.rand((b, model_cfg.task_dim), generator=rng),
        observed_results=torch.zeros((b, t, model_cfg.outcome_dim)),
        result_valid=torch.zeros((b, t, model_cfg.outcome_dim), dtype=torch.bool),
        segment_ids=torch.tensor([[0, 1]], dtype=torch.long),
        anchor_poses=anchors,
        metric_valid=torch.tensor([False]),
        source_ids=torch.tensor([[101, 102]], dtype=torch.long),
        reference_version=torch.tensor([3], dtype=torch.long),
    )

    objects = 1
    instance_masks = torch.zeros((b, t, objects, h, h))
    instance_masks[..., : h // 2, : h // 2] = 1
    targets = TrainingTargets(
        poses=torch.zeros((b, objects, 6)),
        pose_valid=torch.ones((b, objects, 6), dtype=torch.bool),
        instance_masks=instance_masks,
        entity_valid=torch.ones((b, objects), dtype=torch.bool),
        visibility=torch.ones((b, t, objects), dtype=torch.bool),
        change_times=torch.zeros((b, objects, objects), dtype=torch.long),
        change_valid=torch.zeros((b, objects, objects), dtype=torch.bool),
        future_actions=torch.rand((b, model_cfg.action_dim), generator=rng),
        future_response=torch.zeros((b, model_cfg.robot_dim)),
        future_response_valid=torch.ones((b, model_cfg.robot_dim), dtype=torch.bool),
        future_image=torch.zeros((b, c, 3, h, h)),
        future_image_valid=torch.ones((b, c), dtype=torch.bool),
        future_success=torch.ones((b,)),
        future_success_valid=torch.ones((b,), dtype=torch.bool),
    )
    return evidence, targets


def _assert_finite_nonzero_module_grad(module: torch.nn.Module) -> None:
    gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
    assert gradients, "expected a gradient path into this module"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)


def test_two_pass_loss_backward_reaches_conditioner_and_visual_encoder(cfg) -> None:
    evidence, targets = _evidence_and_targets(cfg)
    model = RCHPFN(cfg.model).train()
    output = model(evidence, targets.future_actions)
    assert output.read_bias is not None
    assert output.reconstruction_error is not None

    loss, metrics = ResearchLoss(cfg)(output, evidence, targets)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.tensor(list(metrics.values()))).all()

    (first_pose_grad,) = torch.autograd.grad(
        output.final.mean.square().sum(), output.first.mean, retain_graph=True
    )
    assert torch.isfinite(first_pose_grad).all()
    assert first_pose_grad.abs().sum() > 0

    loss.backward()

    _assert_finite_nonzero_module_grad(model.conditioner)
    _assert_finite_nonzero_module_grad(model.visual)


def test_masked_padded_pixels_do_not_change_inference(cfg) -> None:
    evidence, _ = _evidence_and_targets(cfg)
    model = RCHPFN(cfg.model).eval()
    padded = evidence.pixel_valid.clone()
    padded[:, :, :, : cfg.model.patch_size, : cfg.model.patch_size] = False
    evidence = replace(evidence, pixel_valid=padded)
    changed_images = evidence.images.clone()
    changed_images.masked_fill_(~padded.unsqueeze(3), 1.0)
    changed = replace(evidence, images=changed_images)

    with torch.no_grad():
        baseline_output = model(evidence)
        changed_output = model(changed)

    torch.testing.assert_close(baseline_output.final.mean, changed_output.final.mean)
    torch.testing.assert_close(baseline_output.final.logits, changed_output.final.logits)
    torch.testing.assert_close(baseline_output.slot_masks, changed_output.slot_masks)
    torch.testing.assert_close(baseline_output.reconstruction, changed_output.reconstruction)


def test_future_actions_change_prediction_input_without_changing_state(cfg) -> None:
    evidence, _ = _evidence_and_targets(cfg)
    model = RCHPFN(cfg.model).eval()
    action_a = torch.zeros((1, cfg.model.action_dim))
    action_b = torch.ones_like(action_a)

    with torch.no_grad():
        output_a = model(evidence, action_a)
        output_b = model(evidence, action_b)

    assert output_a.prediction is not None
    assert output_b.prediction is not None
    for belief_a, belief_b in ((output_a.first, output_b.first), (output_a.final, output_b.final)):
        for field in (
            "logits",
            "mean",
            "diagonal",
            "factor",
            "node_features",
            "change_logits",
            "relevance_logits",
            "contact_logits",
        ):
            torch.testing.assert_close(getattr(belief_a, field), getattr(belief_b, field))


def test_evidence_rejects_unarrived_frames_and_duplicate_sources(cfg) -> None:
    evidence, _ = _evidence_and_targets(cfg)
    unarrived = replace(evidence, arrival_times=torch.tensor([[1.1, 1.0]]))
    with pytest.raises(ValueError, match="unarrived sensor observation"):
        unarrived.validate(cfg.model)

    duplicated = replace(evidence, source_ids=torch.tensor([[101, 101]], dtype=torch.long))
    with pytest.raises(ValueError, match="duplicate observation source"):
        duplicated.validate(cfg.model)


def test_invalid_pixel_labels_cannot_change_matching_or_loss(cfg):
    evidence, targets = _evidence_and_targets(cfg)
    valid = evidence.pixel_valid.clone()
    valid[..., :13, :] = False  # deliberately cuts through patch boundaries
    evidence = replace(evidence, pixel_valid=valid)
    model = RCHPFN(cfg.model).eval()
    output = model(evidence, targets.future_actions)
    poisoned = targets.instance_masks.clone()
    poisoned[..., :13, :] = 1 - poisoned[..., :13, :]
    altered = replace(targets, instance_masks=poisoned)
    left = match_entities(output, evidence, targets, cfg.model)
    right = match_entities(output, evidence, altered, cfg.model)
    for original, changed in zip(left, right):
        for a, b in zip(original, changed):
            torch.testing.assert_close(a, b)
    a, _ = ResearchLoss(cfg)(output, evidence, targets)
    b, _ = ResearchLoss(cfg)(output, evidence, altered)
    torch.testing.assert_close(a, b)
