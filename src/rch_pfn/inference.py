"""Fixed-weight inference and append-only sensor history; no control is auto-executed."""

from __future__ import annotations

from dataclasses import fields, replace

import torch

from .config import ModelConfig
from .geometry import pose_to_matrix, relative_transform
from .losses import scales
from .model import RCHPFN
from .schema import EvidenceBatch, ModelOutput, RecorderEvidenceBatch


@torch.no_grad()
def belief_to_dict(output: ModelOutput, evidence: EvidenceBatch, cfg: ModelConfig) -> dict:
    if evidence.images.shape[0] != 1:
        raise ValueError("JSON inference exports one scene at a time")
    belief = output.final
    physical_mean = belief.mean[0] * scales(cfg, belief.mean.device)
    latest = evidence.frame_valid[0].nonzero()[-1, 0]
    visual_transforms = pose_to_matrix(physical_mean)
    anchors = evidence.anchor_poses[0, latest][None].expand(cfg.modes, -1, -1, -1)
    nodes = torch.cat([visual_transforms, anchors], 1)
    edges = output.edge_index
    relations = relative_transform(nodes[:, edges[0]], nodes[:, edges[1]])
    scale_vector = scales(cfg, belief.mean.device).repeat(cfg.slots)
    result = {
        "schema_version": 1,
        "observation_cutoff_s": float(evidence.query_time[0]),
        "latest_exposure_s": float(evidence.timestamps[0, latest]),
        "reference_version": int(evidence.reference_version[0]),
        "coordinate_frame": "robot_base",
        "translation_unit": "m",
        "rotation_unit": "rad",
        "density_coordinates": "fixed xyz + principal local rotvec chart, not SE3 Haar",
        "calibrated": False,
        "control_ready": False,
        "metric_source_available": bool(evidence.metric_valid[0]),
        "next_action": "observe_or_review",
        "slot_ids_are_persistent_entity_ids": False,
        "mode_weights": belief.weights[0].tolist(),
        "node_pose_chart_mean": physical_mean.tolist(),
        "covariance_diagonal": (belief.diagonal[0] * scale_vector.square()).tolist(),
        "covariance_factor": (belief.factor[0] * scale_vector[None, :, None]).tolist(),
        "covariance_convention": "cov(xyz,rotvec)=diag(diagonal)+factor@factor.T; not tangent covariance",
        "edge_index": edges.T.tolist(),
        "relative_transforms": relations.tolist(),
        "change_probabilities": belief.change_logits[0].softmax(-1).tolist(),
        "change_class_order": ["no_event", "before_window"]
        + [f"window_bin_{i}" for i in range(cfg.time_bins)],
        "task_relevance_scores": None,
        "contact_probabilities": None,
        "auxiliary_head_status": "withheld until supervision and calibration provenance are supplied",
        "slot_presence": output.presence_logits[0, latest].sigmoid().tolist(),
        "association_probabilities": output.association_logits[0, latest].softmax(-1).tolist(),
        "source_ids": evidence.source_ids[0, evidence.frame_valid[0]].tolist(),
    }
    if output.prediction is not None:
        result["prediction"] = {key: value[0].tolist() for key, value in output.prediction.items()}
    return result


class StreamingSession:
    """Append actual observations, keep a bounded causal window, recompute belief.

    Entity/reference resets start a new session. This class does not multiply a prior
    posterior by evidence already used. Source IDs prevent double-counting latest frames.
    """

    def __init__(self, model: RCHPFN):
        self.model = model.eval()
        self.evidence: EvidenceBatch | None = None
        self._external_segments: torch.Tensor | None = None

    def append(self, observation: EvidenceBatch) -> None:
        if observation.images.shape[0] != 1:
            raise ValueError("one scene per streaming session")
        temporal_names = {f.name for f in fields(EvidenceBatch)} - {
            "query_time",
            "task",
            "metric_valid",
            "reference_version",
            "recorder",
        }
        keep = observation.frame_valid[0].nonzero().flatten()
        if not keep.numel():
            raise ValueError("cannot append an empty observed prefix")
        observation = replace(
            observation, **{name: getattr(observation, name)[:, keep] for name in temporal_names}
        )
        observation = replace(observation, recorder=_slice_recorder(observation.recorder, keep))
        # Stable external segment IDs can exceed the model's local window capacity.
        original_segments = observation.segment_ids.clone()
        if (
            original_segments.min() < 0
            or (original_segments[:, 1:] < original_segments[:, :-1]).any()
        ):
            raise ValueError("external segment IDs must be chronological and nonnegative")
        _, local = original_segments.unique_consecutive(return_inverse=True)
        observation = replace(observation, segment_ids=local)
        observation.validate(self.model.cfg)
        if self.evidence is None:
            self.evidence = observation
            self._external_segments = original_segments
            return
        previous = self.evidence
        if original_segments[0, 0] < self._external_segments[0, -1]:
            raise ValueError("external segment IDs moved backwards")
        if not torch.equal(previous.reference_version, observation.reference_version):
            raise ValueError("reference changed; create a new session with explicit entity mapping")
        if not torch.equal(previous.task, observation.task):
            raise ValueError("task changed; create a new session")
        if (
            observation.timestamps[0, observation.frame_valid[0]].min()
            < previous.timestamps[0, previous.frame_valid[0]].max()
        ):
            raise ValueError("late data must be aligned into a new causal prefix before append")
        known = set(previous.source_ids[0, previous.frame_valid[0]].tolist())
        if known.intersection(observation.source_ids[0, observation.frame_valid[0]].tolist()):
            raise ValueError("duplicate observed source ID")
        values = {f.name: getattr(observation, f.name) for f in fields(EvidenceBatch)}
        for name in temporal_names:
            values[name] = torch.cat([getattr(previous, name), getattr(observation, name)], 1)
        values["recorder"] = _concat_recorders(
            previous.recorder, observation.recorder, self.model.cfg.recorder_trace_frames
        )
        values["segment_ids"] = torch.cat([self._external_segments, original_segments], 1)
        # Enforce segment and frame budgets, preserving chronological source IDs.
        ids = values["segment_ids"][0]
        unique = ids.unique_consecutive()
        cutoff = unique[max(0, unique.numel() - self.model.cfg.max_segments)]
        keep = (ids >= cutoff).nonzero().flatten()[-self.model.cfg.max_frames :]
        for name in temporal_names:
            values[name] = values[name][:, keep]
        values["recorder"] = _slice_recorder(values["recorder"], keep)
        retained_external_segments = values["segment_ids"].clone()
        _, remapped = values["segment_ids"].unique_consecutive(return_inverse=True)
        values["segment_ids"] = remapped
        result = EvidenceBatch(**values)
        result.validate(self.model.cfg)
        self.evidence = result
        self._external_segments = retained_external_segments

    @torch.no_grad()
    def infer(self, future_actions=None) -> ModelOutput:
        if self.evidence is None:
            raise ValueError("no observations")
        return self.model(self.evidence, future_actions)


def _slice_recorder(
    recorder: RecorderEvidenceBatch | None, keep: torch.Tensor
) -> RecorderEvidenceBatch | None:
    if recorder is None:
        return None
    temporal = {
        "body_positions",
        "body_orientations_wxyz",
        "body_valid",
        "joint_positions",
        "joint_velocities",
        "joint_targets",
        "applied_targets",
        "joint_valid",
        "contact_force_base",
        "contact_normal_force",
        "contact_count",
        "contact_valid",
        "contact_points_base",
        "contact_normals_base",
        "contact_separations",
        "contact_point_valid",
        "phase_ids",
        "phase_valid",
        "sample_time",
        "sample_valid",
    }
    return replace(recorder, **{name: getattr(recorder, name)[:, keep] for name in temporal})


def _concat_recorders(
    left: RecorderEvidenceBatch | None,
    right: RecorderEvidenceBatch | None,
    max_trace_frames: int,
) -> RecorderEvidenceBatch | None:
    temporal = {
        "body_positions",
        "body_orientations_wxyz",
        "body_valid",
        "joint_positions",
        "joint_velocities",
        "joint_targets",
        "applied_targets",
        "joint_valid",
        "contact_force_base",
        "contact_normal_force",
        "contact_count",
        "contact_valid",
        "contact_points_base",
        "contact_normals_base",
        "contact_separations",
        "contact_point_valid",
        "phase_ids",
        "phase_valid",
        "sample_time",
        "sample_valid",
    }
    trace_names = {
        "trace_times",
        "trace_body_positions",
        "trace_body_orientations_wxyz",
        "trace_valid",
    }
    if left is None and right is None:
        return None
    template = left if left is not None else right
    assert template is not None
    left_frames = left.sample_valid.shape[1] if left is not None else 0
    right_frames = right.sample_valid.shape[1] if right is not None else 0
    values: dict[str, torch.Tensor | None] = {}
    for name in temporal:
        sample = getattr(template, name)
        assert sample is not None
        left_value = getattr(left, name) if left is not None else sample.new_zeros((1, left_frames, *sample.shape[2:]))
        right_value = getattr(right, name) if right is not None else sample.new_zeros((1, right_frames, *sample.shape[2:]))
        values[name] = torch.cat([left_value, right_value], dim=1)

    traced = [
        row
        for row in (left, right)
        if row is not None and row.trace_times is not None and row.trace_valid is not None
    ]
    if traced:
        trace_values = {
            name: torch.cat([getattr(row, name) for row in traced], dim=1)
            for name in trace_names
        }
        valid_indices = trace_values["trace_valid"][0].nonzero().flatten()
        if valid_indices.numel():
            times = trace_values["trace_times"][:, valid_indices]
            order = torch.argsort(times[0], stable=True)
            ordered = valid_indices[order]
            ordered_times = trace_values["trace_times"][0, ordered]
            keep = torch.ones_like(ordered_times, dtype=torch.bool)
            keep[:-1] = ordered_times[:-1] != ordered_times[1:]
            ordered = ordered[keep][-max_trace_frames:]
            for name in trace_names:
                values[name] = trace_values[name][:, ordered]
        else:
            values.update({name: None for name in trace_names})
    else:
        values.update({name: None for name in trace_names})
    return RecorderEvidenceBatch(**values)
