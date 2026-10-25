"""Strict, serializable experiment configuration. Unknown keys are errors."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    image_size: int = 224
    patch_size: int = 16
    vision_dim: int = 384
    vision_layers: int = 12
    vision_heads: int = 6
    dim: int = 256
    heads: int = 8
    slots: int = 32
    slot_iterations: int = 3
    relation_layers: int = 2
    core_layers: int = 6
    graph_layers: int = 3
    prediction_layers: int = 2
    predictive_samples: int = 4
    ff_multiplier: int = 4
    modes: int = 8
    covariance_rank: int = 8
    max_cameras: int = 2
    max_frames: int = 132
    max_segments: int = 36
    action_dim: int = 8
    robot_dim: int = 16
    task_dim: int = 16
    outcome_dim: int = 8
    time_bins: int = 16
    dropout: float = 0.0
    bias_bound: float = 4.0
    min_std: float = 0.01
    translation_scale: float = 0.1
    rotation_scale: float = 1.0
    edge_chunk_size: int = 128
    rounds: int = 2
    condition_history: bool = True

    def validate(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, int) and not isinstance(v, bool) and v < 1:
                raise ValueError(f"model.{f.name} must be positive")
        if self.dim % self.heads or self.vision_dim % self.vision_heads:
            raise ValueError("attention widths must be divisible by head counts")
        if self.image_size % self.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        if self.rounds not in (1, 2):
            raise ValueError("rounds must be 1 or 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for name in ("min_std", "translation_scale", "rotation_scale", "bias_bound"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class TrainConfig:
    seed: int = 20261025
    steps: int = 10000
    batch_size: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "cpu"
    num_workers: int = 0
    cpu_threads: int = 4
    ema_decay: float = 0.99
    eval_every: int = 200
    save_every: int = 200
    vision_weight: float = 1.0
    state_weight: float = 1.0
    prediction_weight: float = 1.0
    geometry_weight: float = 0.1
    round1_weight: float = 0.25


@dataclass
class PriorConfig:
    systems: int = 100
    episodes_per_system: int = 4
    frames: int = 8
    min_objects: int = 2
    max_objects: int = 4
    dt: float = 0.1
    position_range: float = 0.12
    rotation_range: float = 0.6
    change_probability: float = 0.6
    occlusion_probability: float = 0.15
    noise_std: float = 0.015
    train_fraction: float = 0.7
    val_fraction: float = 0.15


@dataclass
class Config:
    model: ModelConfig
    train: TrainConfig
    prior: PriorConfig

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        self.model.validate()
        t, p = self.train, self.prior
        if min(t.steps, t.batch_size, t.eval_every, t.save_every) < 1:
            raise ValueError("training counts must be positive")
        if t.num_workers != 0:
            raise ValueError("step-exact sampler currently supports num_workers=0 only")
        if (
            t.cpu_threads < 1
            or not 0 <= t.ema_decay < 1
            or t.learning_rate <= 0
            or t.grad_clip <= 0
        ):
            raise ValueError("invalid optimizer/worker configuration")
        if (
            min(
                t.vision_weight,
                t.state_weight,
                t.prediction_weight,
                t.geometry_weight,
                t.round1_weight,
                t.weight_decay,
            )
            < 0
        ):
            raise ValueError("loss weights and weight decay must be nonnegative")
        if not 1 <= p.min_objects <= p.max_objects <= self.model.slots:
            raise ValueError("prior object count must fit the visual slot budget")
        if not 2 <= p.frames <= self.model.max_frames or p.systems < 3:
            raise ValueError("invalid prior frames/systems")
        if p.episodes_per_system < 1 or p.dt <= 0 or p.noise_std < 0:
            raise ValueError("invalid prior sampling parameters")
        if not 0 < p.train_fraction < 1 or not 0 < p.val_fraction < 1:
            raise ValueError("split fractions must be inside (0, 1)")
        if p.train_fraction + p.val_fraction >= 1:
            raise ValueError("test split must be nonempty")
        for v in (p.change_probability, p.occlusion_probability):
            if not 0 <= v <= 1:
                raise ValueError("prior probabilities must be in [0, 1]")


def config_from_dict(raw: dict[str, Any]) -> Config:
    unknown = set(raw) - {"model", "train", "prior"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    result = Config(
        ModelConfig(**raw.get("model", {})),
        TrainConfig(**raw.get("train", {})),
        PriorConfig(**raw.get("prior", {})),
    )
    result.validate()
    return result


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config_from_dict(raw)
