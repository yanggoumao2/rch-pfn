"""Shared two-pass visual relation inference.

The context/query split prevents queries from writing answers into evidence.
Modes remain separate through conditioning and the second shared graph decoder.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .recorder_encoder import RecorderEncoder
from .schema import EvidenceBatch, JointBelief, ModelOutput


def mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


def encoder(dim: int, heads: int, layers: int, multiplier: int, dropout: float):
    layer = nn.TransformerEncoderLayer(
        dim, heads, dim * multiplier, dropout, activation="gelu", batch_first=True, norm_first=True
    )
    return nn.TransformerEncoder(layer, layers, nn.LayerNorm(dim), enable_nested_tensor=False)


def time_features(times: Tensor) -> Tensor:
    return torch.stack([times, times.sin(), times.cos(), torch.log1p(times.abs())], -1)


class VisualSlots(nn.Module):
    """Frame-local ViT and causal recurrent soft slots with an unexplained patch path."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, v, s = cfg.dim, cfg.vision_dim, cfg.slots
        self.patch = nn.Conv2d(3, v, cfg.patch_size, stride=cfg.patch_size)
        self.positions = nn.Parameter(
            torch.randn(1, (cfg.image_size // cfg.patch_size) ** 2, v) * 0.02
        )
        self.vision = encoder(
            v, cfg.vision_heads, cfg.vision_layers, cfg.ff_multiplier, cfg.dropout
        )
        self.null_patch = nn.Parameter(torch.zeros(1, 1, v))
        self.to_dim = nn.Linear(v, d)
        self.camera_embedding = nn.Embedding(cfg.max_cameras, d)
        self.calibration = mlp(25, d, d)
        self.initial = nn.Parameter(torch.randn(1, s, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.update = nn.GRUCell(d, d)
        self.refine = mlp(d, d * 2, d)
        self.presence = nn.Linear(d, 1)
        self.association_q = nn.Linear(d, d)
        self.association_k = nn.Linear(d, d)
        self.unmatched = nn.Linear(d, 1)
        self.background = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_q = nn.Linear(d, d)
        self.mask_k = nn.Linear(d, d)
        self.pixel_decoder = mlp(d, d * 2, 3 * cfg.patch_size**2)

    def encode_images(
        self, images: Tensor, pixel_valid: Tensor, camera_valid: Tensor
    ) -> tuple[Tensor, Tensor]:
        b, t, c, _, h, w = images.shape
        cfg = self.cfg
        valid_pixels = pixel_valid & camera_valid[..., None, None]
        clean = images * valid_pixels.unsqueeze(3)
        patches = self.patch(clean.reshape(-1, 3, h, w)).flatten(2).transpose(1, 2)
        mask = (
            F.avg_pool2d(
                valid_pixels.float().reshape(-1, 1, h, w), cfg.patch_size, cfg.patch_size
            ).flatten(1)
            > 0
        )
        patches = patches + self.positions
        null = self.null_patch.expand(patches.shape[0], -1, -1)
        augmented = torch.cat([patches, null], 1)
        augmented_mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
        encoded = self.vision(augmented, src_key_padding_mask=~augmented_mask)[:, :-1]
        return self.to_dim(encoded).reshape(b, t, c, -1, cfg.dim), mask.reshape(b, t, c, -1)

    def forward(self, evidence: EvidenceBatch) -> dict[str, Tensor]:
        cfg, d = self.cfg, self.cfg.dim
        b, t, c, _, h, w = evidence.images.shape
        visual_valid = (
            evidence.camera_valid
            & evidence.frame_valid[..., None]
            & evidence.pixel_valid.flatten(-2).any(-1)
        )
        if not visual_valid.any():
            slots = self.initial[:, None].expand(b, t, -1, -1)
            initial = slots.reshape(b * t, cfg.slots, d)
            association = self.association_q(initial) @ self.association_k(initial).transpose(-1, -2)
            association = torch.cat(
                [association / math.sqrt(d), self.unmatched(initial)], -1
            ).reshape(b, t, cfg.slots, cfg.slots + 1)
            return {
                "slots": slots,
                "masks": evidence.images.new_zeros(
                    b, t, cfg.slots + 1, c, (h // cfg.patch_size) ** 2
                ),
                "presence": self.presence(initial).squeeze(-1).reshape(b, t, cfg.slots),
                "association": association,
                "reconstruction": torch.zeros_like(evidence.images),
                "background": self.background.expand(b, t, -1),
            }
        patches, patch_valid = self.encode_images(
            evidence.images,
            evidence.pixel_valid,
            evidence.camera_valid & evidence.frame_valid[..., None],
        )
        b, t, c, p, _ = patches.shape
        cam = self.camera_embedding(torch.arange(c, device=patches.device))
        patches = (
            patches
            + cam[None, None, :, None]
            + self.calibration(evidence.calibration)[..., None, :]
        )
        previous = self.initial.expand(b, -1, -1)
        slots, masks, presences, associations, recons, backgrounds = [], [], [], [], [], []
        for index in range(t):
            context = patches[:, index].reshape(b, c * p, d)
            valid = patch_valid[:, index].reshape(b, c * p)
            # A null entry ensures missing cameras never produce NaN attention.
            context_null = torch.cat([context, self.background.expand(b, -1, -1)], 1)
            valid_null = torch.cat(
                [valid, torch.ones(b, 1, dtype=torch.bool, device=valid.device)], 1
            )
            current = previous
            for _ in range(cfg.slot_iterations):
                read = self.cross(
                    self.norm(current),
                    context_null,
                    context_null,
                    key_padding_mask=~valid_null,
                    need_weights=False,
                )[0]
                current = self.update(read.reshape(-1, d), current.reshape(-1, d)).reshape(b, -1, d)
                current = current + self.refine(self.norm(current))
            assoc = self.association_q(current) @ self.association_k(previous).transpose(-1, -2)
            assoc = torch.cat([assoc / math.sqrt(d), self.unmatched(current)], -1)
            # Recurrent slots carry identity information; unmatched can suppress it.
            memory = assoc.softmax(-1)[..., :-1] @ previous
            current = current + memory
            frame_ok = evidence.frame_valid[:, index] & valid.any(-1)
            current = torch.where(frame_ok[:, None, None], current, previous)
            bg = self.background.expand(b, -1, -1)
            all_slots = torch.cat([current, bg], 1)
            mask_logits = (
                self.mask_q(all_slots) @ self.mask_k(context).transpose(-1, -2) / math.sqrt(d)
            )
            weights = mask_logits.softmax(1)
            reconstructed = weights.transpose(1, 2) @ all_slots
            rgb_patches = self.pixel_decoder(reconstructed).reshape(b * c, p, -1).transpose(1, 2)
            rgb = (
                F.fold(
                    rgb_patches,
                    (cfg.image_size, cfg.image_size),
                    cfg.patch_size,
                    stride=cfg.patch_size,
                )
                .sigmoid()
                .reshape(b, c, 3, cfg.image_size, cfg.image_size)
            )
            bg_weight = weights[:, -1] * valid
            raw_background = (context * bg_weight[..., None]).sum(1) / bg_weight.sum(1).clamp_min(
                1
            )[:, None]
            slots.append(current)
            masks.append(weights.reshape(b, cfg.slots + 1, c, p))
            presences.append(self.presence(current).squeeze(-1))
            associations.append(assoc)
            backgrounds.append(raw_background)
            recons.append(rgb)
            previous = current
        return {
            "slots": torch.stack(slots, 1),
            "masks": torch.stack(masks, 1),
            "presence": torch.stack(presences, 1),
            "association": torch.stack(associations, 1),
            "reconstruction": torch.stack(recons, 1),
            "background": torch.stack(backgrounds, 1),
        }


class ContextBlock(nn.Module):
    """Pre-normalized context self-attention and read-only query cross-attention."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.heads
        self.attention = nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = mlp(d, d * cfg.ff_multiplier, d)

    def context(self, x: Tensor, valid: Tensor) -> Tensor:
        n = self.norm1(x)
        x = x + self.attention(n, n, n, key_padding_mask=~valid, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))

    def query(self, q: Tensor, context: Tensor, valid: Tensor, bias: Tensor | None) -> Tensor:
        n = self.norm1(context)
        mask = torch.zeros(q.shape[0], q.shape[1], context.shape[1], device=q.device, dtype=q.dtype)
        mask = mask.masked_fill(~valid[:, None], float("-inf"))
        if bias is not None:
            mask = mask + bias
        mask = mask.repeat_interleave(self.heads, 0)
        q = q + self.attention(self.norm1(q), n, n, attn_mask=mask, need_weights=False)[0]
        return q + self.ffn(self.norm2(q))


class SharedContextCore(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.blocks = nn.ModuleList([ContextBlock(cfg) for _ in range(cfg.core_layers)])
        self.norm = nn.LayerNorm(cfg.dim)

    def forward(
        self, context: Tensor, valid: Tensor, query: Tensor, bias: Tensor | None = None
    ) -> Tensor:
        for block in self.blocks:
            # Queries cannot alter evidence, including on the next layer.
            query = block.query(query, context, valid, bias)
            context = block.context(context, valid)
        return self.norm(query)


class SceneDecoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.graph = encoder(cfg.dim, cfg.heads, cfg.graph_layers, cfg.ff_multiplier, cfg.dropout)
        self.modes = nn.Parameter(torch.randn(cfg.modes, cfg.dim) * 0.02)
        self.pose = nn.Linear(cfg.dim, 6)
        self.variance = nn.Linear(cfg.dim, 6)
        self.factor = nn.Linear(cfg.dim, 6 * cfg.covariance_rank)
        self.score = nn.Linear(cfg.dim, 1)
        self.relations = mlp(cfg.dim * 2, cfg.dim, cfg.dim)
        self.change = nn.Linear(cfg.dim, cfg.time_bins + 2)
        self.relevance = nn.Linear(cfg.dim, 1)
        self.contact = nn.Linear(cfg.dim, 3)

    def forward(
        self, edge_features: Tensor, latest_nodes: Tensor, background: Tensor, edge_index: Tensor
    ) -> JointBelief:
        # edge_features [B,K,E,D]; both passes call exactly this same decoder.
        b, k, _, d = edge_features.shape
        n, s, cfg = latest_nodes.shape[1], self.cfg.slots, self.cfg
        nodes = latest_nodes[:, None].expand(-1, k, -1, -1).clone()
        aggregate = torch.zeros_like(nodes)
        aggregate.index_add_(2, edge_index[0], edge_features)
        aggregate.index_add_(2, edge_index[1], edge_features)
        nodes = nodes + aggregate / (2 * (n - 1))
        mode = self.modes[None, :k, None].expand(b, -1, -1, -1)
        bg = background[:, None, None].expand(-1, k, -1, -1)
        graph = torch.cat([nodes, mode, bg], 2).reshape(b * k, n + 2, d)
        decoded = self.graph(graph).reshape(b, k, n + 2, d)
        nodes = decoded[:, :, :n]
        means = self.pose(nodes[:, :, :s])
        diagonal = (F.softplus(self.variance(nodes[:, :, :s])) + cfg.min_std).square().flatten(-2)
        factor = self.factor(nodes[:, :, :s]).reshape(b, k, s * 6, cfg.covariance_rank) / math.sqrt(
            cfg.covariance_rank
        )
        relation = self.relations(
            torch.cat([nodes[:, :, edge_index[0]], nodes[:, :, edge_index[1]]], -1)
        )
        return JointBelief(
            self.score(decoded[:, :, n]).squeeze(-1),
            means,
            diagonal,
            factor,
            nodes,
            self.change(relation),
            self.relevance(relation).squeeze(-1),
            self.contact(relation),
        )


class HistoryConditioner(nn.Module):
    """Mode-specific reconstruction/temporal consistency, never a fresh observation."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.dim
        self.cfg = cfg
        self.expected = mlp(d * 2 + cfg.action_dim + 4, d, d)
        self.condition = mlp(d * 3 + 4, d, d + 1)

    def forward(
        self, hypothesis: Tensor, context: Tensor, actions: Tensor, times: Tensor, valid: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        # [B,K,E,2D] hypothesis; [B,E,L,D] original evidence tokens.
        b, k, e, _ = hypothesis.shape
        length, d = context.shape[-2:]
        hyp = hypothesis[..., None, :].expand(-1, -1, -1, length, -1)
        tf = time_features(times)[:, None, None].expand(-1, k, e, -1, -1)
        act = actions[:, None, None].expand(-1, k, e, -1, -1)
        expected = self.expected(torch.cat([hyp, act, tf], -1))
        residual = context[:, None] - expected
        conditioned = self.condition(torch.cat([hyp, residual, tf], -1))
        support = valid[:, None, None, :, None]
        delta = (conditioned[..., :d] * support).sum(-2) / support.sum(-2).clamp_min(1)
        bias = self.cfg.bias_bound * conditioned[..., -1].tanh()
        bias = bias.masked_fill(~valid[:, None, None], 0)
        error = residual.square().mean(-1).masked_fill(~valid[:, None, None], 0)
        return delta, bias, error


class ActionPredictor(nn.Module):
    """Conditional output densities read the same joint belief; no future action in state."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.action = mlp(cfg.action_dim, cfg.dim, cfg.dim)
        self.pose = nn.Linear(6, cfg.dim)
        self.blocks = nn.ModuleList(
            [
                nn.MultiheadAttention(cfg.dim, cfg.heads, cfg.dropout, batch_first=True)
                for _ in range(cfg.prediction_layers)
            ]
        )
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = mlp(cfg.dim, cfg.dim * 2, cfg.robot_dim * 2 + cfg.dim * 2 + 1)

    def forward(self, belief: JointBelief, action: Tensor) -> dict[str, Tensor]:
        b, k, n, d = belief.node_features.shape
        cfg = self.cfg
        # Joint samples preserve low-rank dependence. Fixed quadrature seed at evaluation
        # makes reports reproducible without replacing uncertain states by their means.
        m = cfg.predictive_samples
        generator = (
            None if self.training else torch.Generator(device=action.device).manual_seed(1729)
        )
        shared = torch.randn(
            b, k, m, cfg.covariance_rank, 1, device=action.device, generator=generator
        )
        eps = torch.randn(b, k, m, cfg.slots * 6, device=action.device, generator=generator)
        sample = (
            belief.mean.flatten(-2)[:, :, None]
            + belief.diagonal.sqrt()[:, :, None] * eps
            + (belief.factor[:, :, None] @ shared).squeeze(-1)
        )
        poses = sample.reshape(b, k * m, cfg.slots, 6)
        pose_features = F.pad(self.pose(poses), (0, 0, 0, n - cfg.slots))
        repeated_nodes = (
            belief.node_features[:, :, None].expand(-1, -1, m, -1, -1).reshape(b, k * m, n, d)
        )
        context = (repeated_nodes + pose_features).reshape(b * k * m, n, d)
        q = self.action(action)[:, None, None].expand(-1, k * m, -1, -1).reshape(b * k * m, 1, d)
        for attention in self.blocks:
            q = q + attention(self.norm(q), context, context, need_weights=False)[0]
        values = self.head(self.norm(q[:, 0])).reshape(b, k * m, -1)
        r = cfg.robot_dim
        return {
            "mixture_logits": belief.logits[..., None].expand(-1, -1, m).reshape(b, k * m)
            - math.log(m),
            "response_mean": values[..., :r],
            "response_std": F.softplus(values[..., r : 2 * r]) + cfg.min_std,
            "appearance_mean": values[..., 2 * r : 2 * r + d],
            "appearance_std": F.softplus(values[..., 2 * r + d : 2 * r + 2 * d]) + cfg.min_std,
            "success_logits": values[..., -1],
        }


class RCHPFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        d = cfg.dim
        self.visual = VisualSlots(cfg)
        self.anchor = mlp(16, d, d)
        self.robot = mlp(cfg.robot_dim + cfg.action_dim + 2 * cfg.outcome_dim + 4, d, d)
        self.task = mlp(cfg.task_dim, d, d)
        self.relation = mlp(d * 2, d, d)
        self.relation_quality = mlp(6, d, d)
        self.relation_blocks = nn.ModuleList([mlp(d, d * 2, d) for _ in range(cfg.relation_layers)])
        self.relation_norm = nn.LayerNorm(d)
        self.query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.null_evidence = nn.Parameter(torch.zeros(1, 1, d))
        self.core = SharedContextCore(cfg)
        self.graph = SceneDecoder(cfg)
        self.conditioner = HistoryConditioner(cfg)
        self.hypothesis_state = mlp(12 + cfg.time_bins + 3, d, d * 2)
        self.predictor = ActionPredictor(cfg)
        self.recorder = RecorderEncoder(cfg) if cfg.recorder_enabled else None
        self.recorder_attention = (
            nn.MultiheadAttention(d, cfg.heads, cfg.dropout, batch_first=True)
            if cfg.recorder_enabled
            else None
        )
        self.recorder_null = nn.Parameter(torch.zeros(1, 1, d)) if cfg.recorder_enabled else None
        n = cfg.slots + 2
        edge = torch.ones(n, n, dtype=torch.bool).fill_diagonal_(False).nonzero().T
        self.register_buffer("edge_index", edge, persistent=False)

    def _pool_segments(self, features: Tensor, evidence: EvidenceBatch) -> tuple:
        b, t, e, d = features.shape
        length = int(evidence.segment_ids[evidence.frame_valid].max()) + 1
        weights = F.one_hot(evidence.segment_ids.clamp(0, length - 1), length).to(features.dtype)
        weights = weights * evidence.frame_valid[..., None]
        counts = weights.sum(1)
        valid = counts > 0
        weights = weights / counts[:, None].clamp_min(1)
        context = torch.einsum("btl,bted->beld", weights, features)
        actions = torch.einsum("btl,bta->bla", weights, evidence.actions)
        times = torch.einsum(
            "btl,bt->bl", weights, evidence.timestamps - evidence.query_time[:, None]
        )
        return context, actions, times, valid

    def _read(
        self, context: Tensor, valid: Tensor, queries: Tensor, bias: Tensor | None = None
    ) -> Tensor:
        b, e, length, d = context.shape
        k = queries.shape[1]
        chunks = []
        for start in range(0, e, self.cfg.edge_chunk_size):
            end = min(e, start + self.cfg.edge_chunk_size)
            count = end - start
            ctx = context[:, start:end].reshape(b * count, length, d)
            ok = valid[:, None].expand(-1, count, -1).reshape(b * count, length)
            q = queries[:, :, start:end].permute(0, 2, 1, 3).reshape(b * count, k, d)
            local_bias = (
                None
                if bias is None
                else bias[:, :, start:end].permute(0, 2, 1, 3).reshape(b * count, k, length)
            )
            read = self.core(ctx, ok, q, local_bias)
            chunks.append(read.reshape(b, count, k, d).permute(0, 2, 1, 3))
        return torch.cat(chunks, 2)

    def forward(self, evidence: EvidenceBatch, future_actions: Tensor | None = None) -> ModelOutput:
        if not isinstance(evidence, EvidenceBatch):
            raise TypeError("model requires EvidenceBatch, never a dictionary containing GT")
        evidence.validate(self.cfg)
        cfg, edge = self.cfg, self.edge_index
        visual = self.visual(evidence)
        anchors = self.anchor(evidence.anchor_poses.flatten(-2))
        nodes = torch.cat([visual["slots"], anchors], 2)
        b, t = nodes.shape[:2]
        if evidence.recorder is not None and self.recorder is not None:
            recorder_summary, recorder_tokens, recorder_valid = self.recorder(evidence.recorder)
        else:
            if evidence.recorder is not None:
                raise ValueError("recorder evidence requires model.recorder_enabled=true")
            recorder_summary = nodes.new_zeros(nodes.shape[0], nodes.shape[1], nodes.shape[-1])
            recorder_tokens = nodes.new_zeros(nodes.shape[0], nodes.shape[1], 1, nodes.shape[-1])
            recorder_valid = torch.zeros(
                nodes.shape[0], nodes.shape[1], 1, dtype=torch.bool, device=nodes.device
            )
        # Recorder state informs every node before relation construction. This keeps
        # motion/contact evidence available even when RGB is absent or occluded.
        nodes = nodes + recorder_summary[:, :, None, :]
        relative_times = evidence.timestamps - evidence.query_time[:, None]
        telemetry = torch.cat(
            [
                evidence.actions,
                evidence.robots,
                evidence.observed_results * evidence.result_valid,
                evidence.result_valid.float(),
                time_features(relative_times),
            ],
            -1,
        )
        robot = self.robot(telemetry)
        task = self.task(evidence.task)
        relation = self.relation(torch.cat([nodes[:, :, edge[0]], nodes[:, :, edge[1]]], -1))
        presence = F.pad(visual["presence"].sigmoid(), (0, 2), value=1)
        association = visual["association"].softmax(-1)[..., :-1].diagonal(dim1=-2, dim2=-1)
        association = F.pad(association, (0, 2), value=1)
        anchor_flags = torch.zeros_like(presence)
        anchor_flags[..., cfg.slots :] = 1
        quality = torch.stack(
            [
                presence[..., edge[0]],
                presence[..., edge[1]],
                association[..., edge[0]],
                association[..., edge[1]],
                anchor_flags[..., edge[0]],
                anchor_flags[..., edge[1]],
            ],
            -1,
        )
        relation = relation + self.relation_quality(quality)
        relation = relation + robot[:, :, None] + task[:, None, None]
        edges = edge.shape[1]
        d = relation.shape[-1]
        if self.recorder_attention is None or self.recorder_null is None:
            recorder_read = relation.new_zeros(b, t, edges, d)
        else:
            recorder_tokens = torch.cat(
                [recorder_tokens, self.recorder_null[:, :, None, :].expand(b, t, -1, -1)], -2
            )
            recorder_valid = torch.cat(
                [recorder_valid, torch.ones(b, t, 1, dtype=torch.bool, device=nodes.device)], -1
            )
            relation_query = relation.reshape(b * t, edges, d)
            recorder_memory = recorder_tokens.reshape(b * t, recorder_tokens.shape[2], d)
            recorder_mask = ~recorder_valid.reshape(b * t, recorder_valid.shape[2])
            recorder_read = self.recorder_attention(
                relation_query,
                recorder_memory,
                recorder_memory,
                key_padding_mask=recorder_mask,
                need_weights=False,
            )[0].reshape(b, t, edges, d)
        relation = relation + recorder_read
        for block in self.relation_blocks:
            relation = relation + block(self.relation_norm(relation))
        context, actions, times, valid = self._pool_segments(relation, evidence)
        b, edges, _, d = context.shape
        # Missing history is explicit; a learned prior token is shared by both passes.
        context = torch.cat([context, self.null_evidence[:, None].expand(b, edges, -1, -1)], 2)
        valid = torch.cat([valid, torch.ones(b, 1, dtype=torch.bool, device=valid.device)], 1)
        actions = F.pad(actions, (0, 0, 0, 1))
        times = F.pad(times, (0, 1))
        positions = torch.arange(evidence.frame_valid.shape[1], device=context.device)
        latest_index = positions[None].masked_fill(~evidence.frame_valid, -1).max(-1).values
        latest_nodes = nodes[torch.arange(b, device=context.device), latest_index]
        background = visual["background"][torch.arange(b, device=context.device), latest_index]
        initial_query = self.query.expand(b, edges, -1) + task[:, None]
        first_edges = self._read(context, valid, initial_query[:, None])
        first = self.graph(
            first_edges.expand(-1, cfg.modes, -1, -1), latest_nodes, background, edge
        )
        read_bias = reconstruction_error = None
        final = first
        if cfg.rounds == 2:
            hyp = torch.cat(
                [first.node_features[:, :, edge[0]], first.node_features[:, :, edge[1]]], -1
            )
            pose_chart = F.pad(first.mean, (0, 0, 0, 2))
            explicit_state = torch.cat(
                [
                    pose_chart[:, :, edge[0]],
                    pose_chart[:, :, edge[1]],
                    first.change_logits.softmax(-1),
                    first.weights[:, :, None, None].expand(-1, -1, edges, -1),
                ],
                -1,
            )
            hyp = hyp + self.hypothesis_state(explicit_state)
            delta, read_bias, reconstruction_error = self.conditioner(
                hyp, context, actions, times, valid
            )
            if not cfg.condition_history:
                delta, read_bias = torch.zeros_like(delta), torch.zeros_like(read_bias)
            second_query = initial_query[:, None] + delta
            second_edges = self._read(context, valid, second_query, read_bias)
            final = self.graph(second_edges, latest_nodes, background, edge)
        output = ModelOutput(
            first,
            final,
            edge,
            visual["masks"],
            visual["presence"],
            visual["association"],
            visual["reconstruction"],
            background,
            read_bias,
            reconstruction_error,
            valid,
        )
        if future_actions is not None:
            if (
                future_actions.shape != (b, cfg.action_dim)
                or not torch.isfinite(future_actions).all()
            ):
                raise ValueError("future action shape/values are invalid")
            output.prediction = self.predictor(final, future_actions)
            output.first_prediction = self.predictor(first, future_actions)
        return output

    @torch.no_grad()
    def appearance_target(self, future_images: Tensor, camera_valid: Tensor) -> Tensor:
        b, c, _, h, w = future_images.shape
        patches, valid = self.visual.encode_images(
            future_images[:, None],
            torch.ones(b, 1, c, h, w, dtype=torch.bool, device=future_images.device),
            camera_valid[:, None],
        )
        weighted = patches * valid[..., None]
        return weighted.sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)[:, None]
