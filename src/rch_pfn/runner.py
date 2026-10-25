"""Reproducible training, exact step resume, held-out evaluation and checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock

from .config import Config, config_from_dict
from .data import EpisodeDataset, collate_episodes
from .losses import ResearchLoss, aligned_pose_targets, match_entities, scales
from .model import RCHPFN


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def device_for(name: str) -> torch.device:
    if name == "auto":
        raise ValueError("select cpu or an explicit cuda:N after checking workstation occupancy")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing silent CPU fallback")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment_report() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
    }


def source_fingerprint() -> str:
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_checkpoint(path: str | Path, device="cpu") -> dict:
    # No arbitrary pickle globals accepted. Only use provenance-checked project weights.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") not in (1, 2):
        raise ValueError("unsupported checkpoint format")
    config_from_dict(checkpoint["config"])
    return checkpoint


def load_model_state(model: torch.nn.Module, state: dict[str, torch.Tensor], checkpoint_version: int) -> None:
    """Load legacy PFN weights while allowing the optional recorder modules to start fresh."""
    if checkpoint_version == 1:
        result = model.load_state_dict(state, strict=False)
        allowed_missing = {
            key for key in result.missing_keys
            if key.startswith("recorder.")
            or key.startswith("recorder_attention.")
            or key == "recorder_null"
        }
        unexpected = set(result.unexpected_keys)
        if unexpected or allowed_missing != set(result.missing_keys):
            raise ValueError(
                f"legacy checkpoint state mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        return
    model.load_state_dict(state, strict=True)


def save_checkpoint(path: Path, model, teacher, optimizer, cfg, step, manifest_hash, best) -> None:
    numpy_rng = np.random.get_state()
    payload = {
        "format_version": 2,
        "input_contract_version": 2,
        "recorder_policy": cfg.model.recorder_policy,
        "config": cfg.to_dict(),
        "model": model.state_dict(),
        "teacher": teacher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "manifest_sha256": manifest_hash,
        "best_validation": best,
        "source_sha256": source_fingerprint(),
        "environment": environment_report(),
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "numpy": [
                numpy_rng[0],
                numpy_rng[1].tolist(),
                numpy_rng[2],
                numpy_rng[3],
                numpy_rng[4],
            ],
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        "calibrated": False,
        "training_backend": "see dataset manifest",
        "density_coordinates": "fixed xyz + local rotvec chart; not Haar density",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def teacher_target(teacher, images, camera_valid):
    b, c, _, h, w = images.shape
    patches, valid = teacher.encode_images(
        images[:, None],
        torch.ones(b, 1, c, h, w, dtype=torch.bool, device=images.device),
        camera_valid[:, None],
    )
    return (patches * valid[..., None]).sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)[:, None]


@torch.no_grad()
def evaluate_model(model, teacher, dataset, cfg, device, limit: int | None = None) -> dict:
    model.eval()
    teacher.eval()
    criterion = ResearchLoss(cfg)
    total: dict[str, float] = {}
    examples = 0
    pose_squared, pose_count, success_brier, success_count = 0.0, 0, 0.0, 0
    stop = len(dataset) if limit is None else min(len(dataset), limit)
    if stop < 1:
        raise ValueError("evaluation split is empty")
    start = time.perf_counter()
    # A fixed scene-balanced estimator avoids dependence on padding, MC batch grouping
    # and differing label counts. Training batch size does not change evaluation.
    for offset in range(stop):
        batch = [dataset[offset]]
        evidence, target = collate_episodes(batch)
        evidence, target = evidence.to(device), target.to(device)
        output = model(evidence, target.future_actions)
        appearance = teacher_target(teacher, target.future_image, target.future_image_valid)
        _, metrics = criterion(output, evidence, target, appearance)
        count = len(batch)
        for key, value in metrics.items():
            total[key] = total.get(key, 0) + value * count
        matches = match_entities(output, evidence, target, cfg.model)
        values, mask = aligned_pose_targets(target, matches, cfg.model)
        # MAP-mode point diagnostic only; joint NLL remains primary. Never execute this estimate.
        mode = output.final.weights.argmax(-1)
        mean = output.final.mean[torch.arange(count, device=device), mode]
        errors = (mean - values) * scales(cfg.model, device)
        translation_mask = mask[..., :3]
        pose_squared += float((errors[..., :3].square() * translation_mask).sum())
        pose_count += int(translation_mask.sum())
        pred = output.prediction
        probability = (pred["mixture_logits"].softmax(-1) * pred["success_logits"].sigmoid()).sum(
            -1
        )
        success_brier += float(
            (
                (probability - target.future_success.float()).square() * target.future_success_valid
            ).sum()
        )
        success_count += int(target.future_success_valid.sum())
        examples += count
    report = {
        key: value if key.endswith(("_labels", "_dimensions")) else value / examples
        for key, value in total.items()
    }
    report.update(
        {
            "examples": examples,
            "elapsed_seconds": time.perf_counter() - start,
            "translation_coordinate_rmse_m": (pose_squared / pose_count) ** 0.5
            if pose_count
            else None,
            "success_brier": success_brier / success_count if success_count else None,
            "calibrated": False,
            "physical_closed_loop_evaluation": False,
            "aggregation": "scene-balanced; counts summed; one scene per predictive quadrature",
        }
    )
    return report


def train(
    cfg: Config,
    data_root: Path,
    run_dir: Path,
    resume: Path | None = None,
    init_from: Path | None = None,
) -> dict:
    cfg.validate()
    device = device_for(cfg.train.device)
    # On shared workstations, bounded CPU threads avoid saturating simulator/training hosts.
    torch.set_num_threads(min(cfg.train.cpu_threads, os.cpu_count() or 1))
    if run_dir.exists() and any(run_dir.iterdir()) and resume is None:
        raise FileExistsError("run directory is nonempty; choose a new run or explicit --resume")
    run_dir.mkdir(parents=True, exist_ok=True)
    # OS advisory lock is released on process exit/crash; an inert file is not ownership.
    with FileLock(str(run_dir / ".training.lock"), timeout=0):
        return _train_locked(cfg, data_root, run_dir, device, resume, init_from)


def _train_locked(cfg, data_root, run_dir, device, resume, init_from):
    seed_everything(cfg.train.seed)
    training = EpisodeDataset(data_root, "train", cfg)
    validation = EpisodeDataset(data_root, "val", cfg)
    if len(training) == 0:
        raise ValueError("training split is empty")
    manifest_hash = sha256(data_root / "manifest.json")
    model = RCHPFN(cfg.model).to(device)
    teacher = copy.deepcopy(model.visual).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay
    )
    criterion = ResearchLoss(cfg)
    first_step, best = 0, float("inf")
    if resume and init_from:
        raise ValueError("resume and initialization are distinct; specify only one")
    if init_from:
        checkpoint = load_checkpoint(init_from)
        if config_from_dict(checkpoint["config"]).model != cfg.model:
            raise ValueError("native initialization requires an identical model configuration")
        load_model_state(model, checkpoint["model"], checkpoint["format_version"])
        load_model_state(teacher, checkpoint["teacher"], checkpoint["format_version"])
    if resume:
        checkpoint = load_checkpoint(resume)
        previous = config_from_dict(checkpoint["config"])
        if previous.model != cfg.model or previous.prior != cfg.prior:
            raise ValueError("resume model/prior configuration mismatch")
        for name in (
            "seed",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "grad_clip",
            "vision_weight",
            "state_weight",
            "prediction_weight",
            "geometry_weight",
            "round1_weight",
            "ema_decay",
        ):
            if getattr(previous.train, name) != getattr(cfg.train, name):
                raise ValueError(f"resume changes optimization semantics: {name}")
        if checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("resume dataset changed")
        if checkpoint["source_sha256"] != source_fingerprint():
            raise ValueError("resume source changed; use --init-from for a new experiment")
        load_model_state(model, checkpoint["model"], checkpoint["format_version"])
        load_model_state(teacher, checkpoint["teacher"], checkpoint["format_version"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_step, best = checkpoint["step"], checkpoint["best_validation"]
        rng = checkpoint["rng"]
        torch.set_rng_state(rng["torch"].cpu())
        random.setstate(rng["python"])
        n = rng["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        if rng["cuda"] and device.type == "cuda":
            torch.cuda.set_rng_state_all(rng["cuda"])
    if first_step >= cfg.train.steps:
        raise ValueError("resume checkpoint already reached requested steps")
    write_json(run_dir / "config.json", cfg.to_dict())
    write_json(
        run_dir / "provenance.json",
        {
            "environment": environment_report(),
            "dataset_manifest_sha256": manifest_hash,
            "source_sha256": source_fingerprint(),
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(device),
            "resume": str(resume) if resume else None,
            "initialization": str(init_from) if init_from else "random",
        },
    )
    start = time.perf_counter()
    last_validation = None
    for step in range(first_step, cfg.train.steps):
        model.train()
        # Sampling depends only on seed and step, so the resumed loader has the exact next batch.
        generator = torch.Generator().manual_seed(cfg.train.seed + step)
        indices = torch.randint(
            len(training), (cfg.train.batch_size,), generator=generator
        ).tolist()
        evidence, target = collate_episodes([training[index] for index in indices])
        evidence, target = evidence.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(evidence, target.future_actions)
        appearance = teacher_target(teacher, target.future_image, target.future_image_valid)
        loss, metrics = criterion(output, evidence, target, appearance)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite loss at step {step + 1}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.train.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            for ema, live in zip(teacher.parameters(), model.visual.parameters(), strict=True):
                ema.lerp_(live, 1 - cfg.train.ema_decay)
        metrics.update(
            {
                "step": step + 1,
                "gradient_norm": float(grad_norm),
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        with (run_dir / "train.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        if (step + 1) % cfg.train.eval_every == 0 or step + 1 == cfg.train.steps:
            last_validation = evaluate_model(model, teacher, validation, cfg, device)
            write_json(run_dir / f"validation-{step + 1:06d}.json", last_validation)
            if last_validation["loss"] < best:
                best = last_validation["loss"]
                save_checkpoint(
                    run_dir / "best.pt",
                    model,
                    teacher,
                    optimizer,
                    cfg,
                    step + 1,
                    manifest_hash,
                    best,
                )
        if (step + 1) % cfg.train.save_every == 0 or step + 1 == cfg.train.steps:
            save_checkpoint(
                run_dir / "last.pt", model, teacher, optimizer, cfg, step + 1, manifest_hash, best
            )
        print(json.dumps({"step": step + 1, "loss": metrics["loss"]}), flush=True)
    summary = {
        "completed_steps": cfg.train.steps,
        "validation": last_validation,
        "elapsed_seconds": time.perf_counter() - start,
        "device": str(device),
        "checkpoint_sha256": sha256(run_dir / "last.pt"),
        "purpose": "engineering verification unless a research protocol states otherwise",
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def evaluate(checkpoint_path: Path, data_root: Path, split: str, device_name: str) -> dict:
    checkpoint = load_checkpoint(checkpoint_path)
    cfg = config_from_dict(checkpoint["config"])
    torch.set_num_threads(min(cfg.train.cpu_threads, os.cpu_count() or 1))
    device = device_for(device_name)
    model = RCHPFN(cfg.model).to(device)
    load_model_state(model, checkpoint["model"], checkpoint["format_version"])
    teacher = copy.deepcopy(model.visual).eval().requires_grad_(False)
    load_model_state(teacher, checkpoint["teacher"], checkpoint["format_version"])
    dataset = EpisodeDataset(data_root, split, cfg)
    report = evaluate_model(model, teacher, dataset, cfg, device)
    report.update(
        {
            "split": split,
            "checkpoint_sha256": sha256(checkpoint_path),
            "dataset_manifest_sha256": sha256(data_root / "manifest.json"),
            "source_sha256": source_fingerprint(),
        }
    )
    return report
