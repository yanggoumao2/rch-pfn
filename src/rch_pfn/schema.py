"""Sensor-only model inputs and separately typed privileged training targets."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import torch
from torch import Tensor

from .config import ModelConfig


class TensorRecord:
    def to(self, device: str | torch.device):
        return replace(
            self,
            **{
                f.name: getattr(self, f.name).to(device)
                if isinstance(getattr(self, f.name), (Tensor, TensorRecord))
                else getattr(self, f.name)
                for f in fields(self)
                if isinstance(getattr(self, f.name), (Tensor, TensorRecord))
            },
        )


@dataclass
class RecorderEvidenceBatch(TensorRecord):
    """Normalized LabUtopia recorder channels aligned to the model frame window."""

    body_positions: Tensor
    body_orientations_wxyz: Tensor
    body_valid: Tensor
    joint_positions: Tensor
    joint_velocities: Tensor
    joint_targets: Tensor
    applied_targets: Tensor
    joint_valid: Tensor
    contact_force_base: Tensor
    contact_normal_force: Tensor
    contact_count: Tensor
    contact_valid: Tensor
    contact_points_base: Tensor
    contact_normals_base: Tensor
    contact_separations: Tensor
    contact_point_valid: Tensor
    phase_ids: Tensor
    phase_valid: Tensor
    sample_time: Tensor
    sample_valid: Tensor
    trace_times: Tensor | None = None
    trace_body_positions: Tensor | None = None
    trace_body_orientations_wxyz: Tensor | None = None
    trace_valid: Tensor | None = None

    def validate(self, cfg: ModelConfig, batch_size: int, frame_count: int) -> None:
        b, t = batch_size, frame_count
        expected = {
            "body_positions": (b, t, cfg.recorder_body_count, 3),
            "body_orientations_wxyz": (b, t, cfg.recorder_body_count, 4),
            "body_valid": (b, t, cfg.recorder_body_count),
            "joint_positions": (b, t, cfg.recorder_joint_count),
            "joint_velocities": (b, t, cfg.recorder_joint_count),
            "joint_targets": (b, t, cfg.recorder_joint_count),
            "applied_targets": (b, t, cfg.recorder_joint_count),
            "joint_valid": (b, t, cfg.recorder_joint_count),
            "contact_force_base": (b, t, cfg.recorder_contact_pairs, 3),
            "contact_normal_force": (b, t, cfg.recorder_contact_pairs),
            "contact_count": (b, t, cfg.recorder_contact_pairs),
            "contact_valid": (b, t, cfg.recorder_contact_pairs),
            "contact_points_base": (
                b, t, cfg.recorder_contact_pairs, cfg.recorder_contact_points, 3
            ),
            "contact_normals_base": (
                b, t, cfg.recorder_contact_pairs, cfg.recorder_contact_points, 3
            ),
            "contact_separations": (
                b, t, cfg.recorder_contact_pairs, cfg.recorder_contact_points
            ),
            "contact_point_valid": (
                b, t, cfg.recorder_contact_pairs, cfg.recorder_contact_points
            ),
            "phase_ids": (b, t),
            "phase_valid": (b, t),
            "sample_time": (b, t),
            "sample_valid": (b, t),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"recorder.{name}: expected {shape}, got {tuple(value.shape)}")
        for name in (
            "body_valid", "joint_valid", "contact_valid", "contact_point_valid",
            "phase_valid", "sample_valid",
        ):
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"recorder.{name} must be boolean")
        if self.phase_ids.dtype != torch.long or self.contact_count.dtype != torch.long:
            raise ValueError("recorder categorical/count fields must be int64")
        if (self.contact_count < 0).any():
            raise ValueError("recorder.contact_count cannot be negative")
        if cfg.recorder_policy == "deployable":
            privileged_masks = [
                self.body_valid.any(),
                self.contact_valid.any(),
                self.contact_point_valid.any(),
                self.phase_valid.any(),
            ]
            if self.trace_valid is not None:
                privileged_masks.append(self.trace_valid.any())
            if any(bool(mask) for mask in privileged_masks):
                raise ValueError(
                    "deployable recorder policy forbids simulator body, contact, phase, and trace inputs"
                )
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"recorder.{field.name} contains nonfinite values")
        if self.trace_times is not None:
            if self.trace_body_positions is None or self.trace_body_orientations_wxyz is None:
                raise ValueError("recorder trace fields must be supplied as a complete group")
            if self.trace_valid is None:
                raise ValueError("recorder.trace_valid is required with trace fields")
            h = self.trace_times.shape[1]
            trace_shapes = {
                "trace_times": (b, h),
                "trace_body_positions": (b, h, 2, 3),
                "trace_body_orientations_wxyz": (b, h, 2, 4),
                "trace_valid": (b, h),
            }
            for name, shape in trace_shapes.items():
                if tuple(getattr(self, name).shape) != shape:
                    raise ValueError(f"recorder.{name}: expected {shape}")
            if self.trace_valid.dtype != torch.bool:
                raise ValueError("recorder.trace_valid must be boolean")
            if cfg.recorder_trace_frames and h > cfg.recorder_trace_frames:
                raise ValueError("recorder trace exceeds configured capacity")
            for row in range(b):
                times = self.trace_times[row, self.trace_valid[row]]
                if times.numel() and (times[1:] < times[:-1]).any():
                    raise ValueError("recorder trace must be chronological")


@dataclass
class EvidenceBatch(TensorRecord):
    images: Tensor
    pixel_valid: Tensor
    frame_valid: Tensor
    camera_valid: Tensor
    timestamps: Tensor
    arrival_times: Tensor
    query_time: Tensor
    actions: Tensor
    robots: Tensor
    calibration: Tensor
    task: Tensor
    observed_results: Tensor
    result_valid: Tensor
    segment_ids: Tensor
    anchor_poses: Tensor
    metric_valid: Tensor
    source_ids: Tensor
    reference_version: Tensor
    recorder: RecorderEvidenceBatch | None = None

    def validate(self, cfg: ModelConfig) -> None:
        if self.images.ndim != 6:
            raise ValueError("images must have shape [B,T,C,3,H,W]")
        b, t, c, channels, h, w = self.images.shape
        if min(b, t, c) < 1 or channels != 3 or h != cfg.image_size or w != h:
            raise ValueError("image shape does not match configuration")
        if t > cfg.max_frames or c > cfg.max_cameras:
            raise ValueError("evidence capacity exceeded; explicitly select a causal window")
        shapes = {
            "pixel_valid": (b, t, c, h, w),
            "frame_valid": (b, t),
            "camera_valid": (b, t, c),
            "timestamps": (b, t),
            "arrival_times": (b, t),
            "query_time": (b,),
            "actions": (b, t, cfg.action_dim),
            "robots": (b, t, cfg.robot_dim),
            "calibration": (b, t, c, 25),
            "task": (b, cfg.task_dim),
            "observed_results": (b, t, cfg.outcome_dim),
            "result_valid": (b, t, cfg.outcome_dim),
            "segment_ids": (b, t),
            "anchor_poses": (b, t, 2, 4, 4),
            "metric_valid": (b,),
            "source_ids": (b, t),
            "reference_version": (b,),
        }
        for name, expected in shapes.items():
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name}: expected {expected}, got {tuple(value.shape)}")
        bool_fields = ("pixel_valid", "frame_valid", "camera_valid", "metric_valid", "result_valid")
        for name in bool_fields:
            if getattr(self, name).dtype != torch.bool:
                raise ValueError(f"{name} must be boolean")
        for name in ("segment_ids", "source_ids", "reference_version"):
            if getattr(self, name).dtype != torch.long:
                raise ValueError(f"{name} must be int64")
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
                raise ValueError(f"{f.name} contains nonfinite values; use masks, not NaN")
        if self.images.min() < 0 or self.images.max() > 1:
            raise ValueError("RGB images must be in [0,1]")
        valid = self.frame_valid
        if not valid.any(-1).all():
            raise ValueError("each item needs at least one observed frame")
        if ((self.timestamps > self.query_time[:, None]) & valid).any():
            raise ValueError("future exposure in current-state evidence")
        if ((self.arrival_times > self.query_time[:, None]) & valid).any():
            raise ValueError("unarrived sensor observation in evidence")
        if ((self.arrival_times < self.timestamps) & valid).any():
            raise ValueError("arrival cannot precede exposure")
        for row in range(b):
            times, ids = self.timestamps[row, valid[row]], self.source_ids[row, valid[row]]
            segments = self.segment_ids[row, valid[row]]
            if (times[1:] < times[:-1]).any() or (segments[1:] < segments[:-1]).any():
                raise ValueError("evidence must be chronological")
            if ids.unique().numel() != ids.numel():
                raise ValueError("duplicate observation source; deduplicate latest/context")
            if segments.min() < 0 or segments.max() >= cfg.max_segments:
                raise ValueError("segment index out of configured bounds")
            has_visual = (
                self.camera_valid[row, valid[row]]
                & self.pixel_valid[row, valid[row]].flatten(-2).any(-1)
            ).any()
            has_recorder = self.recorder is not None and self.recorder.sample_valid[row].any()
            if not has_visual and not has_recorder:
                raise ValueError("each item needs at least one valid visual observation")
        if self.recorder is not None:
            self.recorder.validate(cfg, b, t)
            if (self.recorder.sample_valid & ~self.frame_valid).any():
                raise ValueError("recorder sample_valid cannot exceed evidence frame_valid")
            aligned = self.recorder.sample_valid & self.frame_valid
            if not torch.allclose(
                self.recorder.sample_time[aligned], self.timestamps[aligned], rtol=0, atol=1e-5
            ):
                raise ValueError("recorder samples must be aligned to evidence timestamps")
            if ((self.recorder.sample_time > self.query_time[:, None]) & self.recorder.sample_valid).any():
                raise ValueError("future recorder sample in current-state evidence")
            if self.recorder.sample_valid.shape != self.frame_valid.shape:
                raise ValueError("recorder frames must be aligned to the evidence frame axis")
            if self.recorder.trace_times is not None and (
                (self.recorder.trace_times > self.query_time[:, None]) & self.recorder.trace_valid
            ).any():
                raise ValueError("future recorder trace sample in current-state evidence")


@dataclass
class TrainingTargets(TensorRecord):
    poses: Tensor
    pose_valid: Tensor
    instance_masks: Tensor
    entity_valid: Tensor
    visibility: Tensor
    change_times: Tensor
    change_valid: Tensor
    future_actions: Tensor
    future_response: Tensor
    future_response_valid: Tensor
    future_image: Tensor
    future_image_valid: Tensor
    future_success: Tensor
    future_success_valid: Tensor
    equivalent_poses: Tensor | None = None
    equivalent_valid: Tensor | None = None
    image_points: Tensor | None = None  # [B,N,2], reference origin pixels, camera 0
    image_points_valid: Tensor | None = None  # [B,N], never inferred from mask centroid
    relation_relevance: Tensor | None = None  # [B,N,N], measured intervention/task target [0,1]
    relation_relevance_valid: Tensor | None = None
    contact_states: Tensor | None = None  # [B,N,N], 0 unknown / 1 no / 2 yes
    contact_valid: Tensor | None = None


@dataclass
class JointBelief:
    """Density in a fixed xyz/rotation-vector chart; modes share low-rank covariance."""

    logits: Tensor  # [B,K]
    mean: Tensor  # [B,K,S,6], normalized chart
    diagonal: Tensor  # [B,K,S*6], strictly positive variance
    factor: Tensor  # [B,K,S*6,rank]
    node_features: Tensor  # [B,K,S+2,D]
    change_logits: Tensor  # [B,K,E,time_bins+2]
    relevance_logits: Tensor  # [B,K,E]
    contact_logits: Tensor  # [B,K,E,3], unknown/no/yes

    @property
    def weights(self) -> Tensor:
        return self.logits.softmax(-1)


@dataclass
class ModelOutput:
    first: JointBelief
    final: JointBelief
    edge_index: Tensor
    slot_masks: Tensor  # [B,T,S+1,C,P], includes background
    presence_logits: Tensor  # [B,T,S]
    association_logits: Tensor  # [B,T,S,S+1], last=new/unmatched
    reconstruction: Tensor  # [B,T,C,3,H,W]
    background: Tensor  # [B,D]
    read_bias: Tensor | None  # [B,K,E,L]
    reconstruction_error: Tensor | None  # [B,K,E,L]
    evidence_mask: Tensor  # [B,L]
    prediction: dict[str, Tensor] | None = None
    first_prediction: dict[str, Tensor] | None = None
