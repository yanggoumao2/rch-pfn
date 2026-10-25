"""Structured fusion of the normalized LabUtopia recorder channels."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .schema import RecorderEvidenceBatch


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class RecorderEncoder(nn.Module):
    """Encode body, joint, contact and trace streams as per-frame evidence tokens.

    The raw record remains available in the NPZ archive. This module consumes the
    normalized, masked view and keeps different physical channels distinct until
    they are fused by attention in the main model.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.body = _mlp(3 + 4 + 1, d, d)
        self.joint = _mlp(4 + 1, d, d)
        self.contact = _mlp(3 + 1 + 1 + 3 + 3 + 1 + 1, d, d)
        self.phase = nn.Embedding(64, d)
        self.phase_valid = nn.Linear(1, d)
        self.trace = _mlp(2 * (3 + 4) + 1 + 1, d, d)
        self.body_id = nn.Embedding(cfg.recorder_body_count, d)
        self.joint_id = nn.Embedding(cfg.recorder_joint_count, d)
        self.contact_id = nn.Embedding(cfg.recorder_contact_pairs, d)
        self.modality = nn.Embedding(5, d)
        self.summary = _mlp(d * 5, d * 2, d)

    def forward(self, recorder: RecorderEvidenceBatch) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.cfg
        b, t = recorder.sample_valid.shape
        device = recorder.sample_valid.device
        body_valid = recorder.body_valid & recorder.sample_valid[..., None]
        body_quaternion = recorder.body_orientations_wxyz / recorder.body_orientations_wxyz.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        body_position = recorder.body_positions / cfg.translation_scale
        body_input = torch.cat(
            [body_position, body_quaternion, body_valid[..., None].float()], -1
        )
        body = self.body(body_input) + self.body_id.weight[None, None]
        body = body * body_valid[..., None]

        joint_valid = recorder.joint_valid & recorder.sample_valid[..., None]
        position_scale = recorder.joint_positions.new_full((cfg.recorder_joint_count,), cfg.rotation_scale)
        velocity_scale = recorder.joint_positions.new_full((cfg.recorder_joint_count,), cfg.recorder_velocity_scale)
        position_scale[..., -2:] = cfg.translation_scale
        velocity_scale[..., -2:] = cfg.translation_scale
        joint_input = torch.stack(
            [
                recorder.joint_positions / position_scale,
                recorder.joint_velocities / velocity_scale,
                recorder.joint_targets / position_scale,
                recorder.applied_targets / position_scale,
                joint_valid.float(),
            ],
            -1,
        )
        joint = self.joint(joint_input)
        joint = joint + self.joint_id.weight[None, None]
        joint = joint * joint_valid[..., None]

        point_valid = recorder.contact_point_valid & recorder.contact_valid[..., None]
        point_count = point_valid.sum(-1).clamp_min(1).to(recorder.contact_points_base.dtype)
        point_mean = (
            recorder.contact_points_base * point_valid[..., None]
        ).sum(-2) / point_count[..., None]
        normal_mean = (recorder.contact_normals_base * point_valid[..., None]).sum(-2) / point_count[..., None]
        separation_mean = (recorder.contact_separations * point_valid).sum(-1) / point_count
        contact_input = torch.cat(
            [recorder.contact_force_base / cfg.recorder_force_scale,
             recorder.contact_normal_force[..., None],
             recorder.contact_count[..., None].float(),
             point_mean / cfg.translation_scale, normal_mean,
             (separation_mean / cfg.translation_scale)[..., None],
             recorder.contact_valid[..., None].float()], -1
        )
        contact = self.contact(contact_input) + self.contact_id.weight[None, None]
        contact = contact * (recorder.contact_valid & recorder.sample_valid[..., None])[..., None]

        phase_id = recorder.phase_ids.clamp(0, self.phase.num_embeddings - 1)
        phase = self.phase(phase_id) + self.phase_valid(recorder.phase_valid[..., None].float())
        phase = phase * (recorder.phase_valid & recorder.sample_valid)[..., None]

        if recorder.trace_times is not None:
            trace_valid = recorder.trace_valid
            trace_time_origin = recorder.trace_times[..., :1]
            trace_input = torch.cat(
                [
                    (recorder.trace_body_positions / cfg.translation_scale).flatten(-2),
                    recorder.trace_body_orientations_wxyz.flatten(-2),
                    ((recorder.trace_times - trace_time_origin) / 1.0)[..., None],
                    trace_valid[..., None].float(),
                ],
                -1,
            )
            trace_points = self.trace(trace_input)
            # A model frame reads only trace samples that were available by its
            # timestamp. Prefix pooling preserves causal multi-rate alignment.
            causal = (
                trace_valid[:, None, :]
                & (recorder.trace_times[:, None, :] <= recorder.sample_time[:, :, None])
            )
            trace_weights = causal.to(trace_points.dtype)
            trace = torch.einsum("bth,bhd->btd", trace_weights, trace_points)
            trace = trace / trace_weights.sum(-1, keepdim=True).clamp_min(1.0)
            trace_valid_frame = causal.any(-1) & recorder.sample_valid
        else:
            trace = recorder.sample_valid[..., None].float() * 0.0
            trace = trace.expand(b, t, self.cfg.dim)
            trace_valid_frame = torch.zeros((b, t), dtype=torch.bool, device=device)

        body_summary = body.sum(-2) / body_valid.sum(-1, keepdim=True).clamp_min(1)
        joint_summary = joint.sum(-2) / joint_valid.sum(-1, keepdim=True).clamp_min(1)
        contact_valid = recorder.contact_valid & recorder.sample_valid[..., None]
        contact_summary = contact.sum(-2) / contact_valid.sum(-1, keepdim=True).clamp_min(1)
        phase_summary = phase
        trace_summary = trace
        summary = self.summary(torch.cat([body_summary, joint_summary, contact_summary, phase_summary, trace_summary], -1))
        summary = summary * recorder.sample_valid[..., None]

        tokens = torch.cat([body, joint, contact, phase[..., None, :], trace[..., None, :]], -2)
        valid = torch.cat([
            body_valid,
            joint_valid,
            recorder.contact_valid & recorder.sample_valid[..., None],
            (recorder.phase_valid & recorder.sample_valid)[..., None],
            trace_valid_frame[..., None],
        ], -1)
        modality_ids = torch.cat([
            torch.zeros(cfg.recorder_body_count, dtype=torch.long, device=device),
            torch.ones(cfg.recorder_joint_count, dtype=torch.long, device=device),
            torch.full((cfg.recorder_contact_pairs,), 2, dtype=torch.long, device=device),
            torch.tensor([3, 4], dtype=torch.long, device=device),
        ])
        tokens = tokens + self.modality(modality_ids)[None, None]
        return summary, tokens, valid
