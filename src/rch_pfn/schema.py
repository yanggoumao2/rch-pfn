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
                for f in fields(self)
                if isinstance(getattr(self, f.name), Tensor)
            },
        )


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
            if value.is_floating_point() and not torch.isfinite(value).all():
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
            if not (
                self.camera_valid[row, valid[row]]
                & self.pixel_valid[row, valid[row]].flatten(-2).any(-1)
            ).any():
                raise ValueError("each item needs at least one valid visual observation")


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
