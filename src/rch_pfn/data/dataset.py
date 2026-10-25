"""Manifest validation, episode loading, collation, and recorder NPZ I/O."""

from __future__ import annotations

import json
import re
from dataclasses import fields, is_dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from ..config import Config, config_from_dict
from ..schema import EvidenceBatch, TrainingTargets
from .format import (
    FORMAT_VERSION,
    OPTIONAL_TARGET_FIELDS,
    SENSOR_FIELDS,
    SPLITS,
    TARGET_FIELDS,
    build_record_arrays,
    file_sha256,
    read_npz,
    safe_identifier,
    split_record_arrays,
    validate_episode_arrays,
    write_npz,
)

MANIFEST_VERSION = 1
SYNTHETIC_BACKEND = "procedural_rgb_kinematic"
RECORDER_BACKEND = "recorder_npz_import"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _episode_path(system_id: str, episode_id: str) -> str:
    return f"episodes/{system_id}/{episode_id}.npz"


def _record_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("episode paths must be portable POSIX relative paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError("episode path escapes the dataset root")
    if len(path.parts) != 3 or path.parts[0] != "episodes" or path.suffix != ".npz":
        raise ValueError("episode path must be episodes/<system>/<episode>.npz")
    resolved_root = root.resolve()
    resolved_file = (root / Path(*path.parts)).resolve()
    if resolved_root not in resolved_file.parents:
        raise ValueError("episode path escapes the dataset root")
    return resolved_file


def make_manifest(
    config: Config,
    *,
    backend: str,
    system_splits: Mapping[str, Sequence[str]],
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_VERSION,
        "generator": backend,
        "backend": backend,
        "config": config.to_dict(),
        "system_splits": {split: sorted(system_splits.get(split, ())) for split in SPLITS},
        "episodes": [dict(item) for item in episodes],
    }


def write_manifest(root: str | Path, manifest: Mapping[str, Any]) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    temporary = root / ".manifest.json.tmp"
    payload = (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_manifest(root: str | Path, config: Config | None = None) -> dict[str, Any]:
    """Validate every hash, source identity, record schema, and system-level split."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read dataset manifest at {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_VERSION:
        raise ValueError("unsupported or malformed dataset manifest")
    backend = manifest.get("backend")
    if not isinstance(backend, str) or manifest.get("generator") != backend:
        raise ValueError("manifest generator/backend mismatch")
    if backend not in (SYNTHETIC_BACKEND, RECORDER_BACKEND):
        raise ValueError(f"unsupported dataset backend: {backend}")
    manifest_config = manifest.get("config")
    if not isinstance(manifest_config, dict):
        raise ValueError("manifest is missing its serialized config")
    effective_config = config if config is not None else config_from_dict(manifest_config)
    effective_config.validate()

    groups = manifest.get("system_splits")
    if not isinstance(groups, dict) or set(groups) != set(SPLITS):
        raise ValueError("system_splits must contain train, val, and test")
    system_split: dict[str, str] = {}
    for split in SPLITS:
        members = groups[split]
        if not isinstance(members, list):
            raise ValueError(f"system_splits.{split} must be a list")
        if len(set(members)) != len(members):
            raise ValueError(f"duplicate system in {split} split")
        for system_id in members:
            safe_identifier(system_id, "system_id")
            if system_id in system_split:
                raise ValueError(f"system {system_id} appears in multiple splits")
            system_split[system_id] = split

    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("manifest must list at least one episode")
    seen_ids: set[tuple[str, str]] = set()
    seen_paths: set[str] = set()
    for item in episodes:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "split",
            "system_id",
            "episode_id",
            "sha256",
        }:
            raise ValueError("malformed manifest episode entry")
        system_id = safe_identifier(item["system_id"], "system_id")
        episode_id = safe_identifier(item["episode_id"], "episode_id")
        split = item["split"]
        if split not in SPLITS or system_split.get(system_id) != split:
            raise ValueError("episode split disagrees with system-level split")
        expected_path = _episode_path(system_id, episode_id)
        if item["path"] != expected_path:
            raise ValueError("episode path does not match its source identity")
        if item["path"] in seen_paths or (system_id, episode_id) in seen_ids:
            raise ValueError("duplicate episode path or source identity")
        seen_paths.add(item["path"])
        seen_ids.add((system_id, episode_id))
        if not isinstance(item["sha256"], str) or not _SHA256.fullmatch(item["sha256"]):
            raise ValueError("malformed episode SHA-256")
        path = _record_path(root, item["path"])
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"episode hash mismatch: {item['path']}")
        sensor, targets, meta = split_record_arrays(read_npz(path))
        if (meta["system_id"], meta["episode_id"], meta["split"], meta["backend"]) != (
            system_id,
            episode_id,
            split,
            backend,
        ):
            raise ValueError(f"record source identity mismatch: {item['path']}")
        validate_episode_arrays(sensor, targets, effective_config)

    episode_systems = {item["system_id"] for item in episodes}
    if episode_systems != set(system_split):
        raise ValueError("manifest system split groups and episode systems differ")
    return manifest


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _sensor_tensors(sensor: Mapping[str, np.ndarray]) -> EvidenceBatch:
    bool_fields = {"pixel_valid", "frame_valid", "camera_valid", "result_valid", "metric_valid"}
    long_fields = {"segment_ids", "source_ids", "reference_version"}
    scalar_fields = {"query_time", "metric_valid", "reference_version"}
    values: dict[str, Tensor] = {}
    for name in SENSOR_FIELDS:
        array = np.asarray(sensor[name])
        tensor = torch.from_numpy(np.array(array, copy=True))
        if name == "images":
            tensor = tensor.float()
            if array.dtype.kind in "ui":
                tensor = tensor / 255.0
        elif name in bool_fields:
            tensor = tensor.bool()
        elif name in long_fields:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        values[name] = tensor.unsqueeze(0) if name not in scalar_fields else tensor.reshape(1)
    return EvidenceBatch(**values)


def _target_tensors(targets: Mapping[str, np.ndarray]) -> TrainingTargets:
    bool_fields = {
        "pose_valid",
        "instance_masks",
        "entity_valid",
        "change_valid",
        "future_response_valid",
        "future_image_valid",
        "future_success_valid",
        "equivalent_valid",
        "image_points_valid",
        "relation_relevance_valid",
        "contact_valid",
    }
    long_fields = {"change_times", "contact_states"}
    image_fields = {"future_image"}
    values: dict[str, Tensor | None] = {}
    for name, raw in targets.items():
        array = np.asarray(raw)
        tensor = torch.from_numpy(np.array(array, copy=True))
        if name in image_fields:
            tensor = tensor.float()
            if array.dtype.kind in "ui":
                tensor = tensor / 255.0
        elif name in bool_fields:
            tensor = tensor.bool()
        elif name in long_fields:
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        values[name] = tensor.unsqueeze(0)
    for name in OPTIONAL_TARGET_FIELDS:
        values.setdefault(name, None)
    return TrainingTargets(**values)  # type: ignore[arg-type]


class EpisodeDataset(torch.utils.data.Dataset):
    """A validated split yielding sensor-only EvidenceBatch and separate targets."""

    def __init__(self, root: str | Path, split: str, config: Config):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        self.root = Path(root)
        self.config = config
        # Full hash and source validation is intentionally part of loader construction.
        self.manifest = validate_manifest(self.root, config)
        self.records = [item for item in self.manifest["episodes"] if item["split"] == split]
        if not self.records:
            raise ValueError(f"dataset split {split!r} is empty")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[EvidenceBatch, TrainingTargets]:
        item = self.records[index]
        path = _record_path(self.root, item["path"])
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"episode changed after manifest validation: {item['path']}")
        sensor, targets, meta = split_record_arrays(read_npz(path))
        if (meta["system_id"], meta["episode_id"], meta["split"]) != (
            item["system_id"],
            item["episode_id"],
            item["split"],
        ):
            raise ValueError("record identity changed after manifest validation")
        validate_episode_arrays(sensor, targets, self.config)
        evidence, supervision = _sensor_tensors(sensor), _target_tensors(targets)
        evidence.validate(self.config.model)
        return evidence, supervision


def _pad_time(values: Sequence[Tensor], max_t: int) -> Tensor:
    first = values[0]
    output = first.new_zeros((len(values), max_t, *first.shape[2:]))
    for index, value in enumerate(values):
        if value.shape[0] != 1 or value.device != first.device or value.dtype != first.dtype:
            raise ValueError("collate expects single-item tensors with matching device and dtype")
        output[index, : value.shape[1]] = value[0]
    return output


def _pad_entities(values: Sequence[Tensor], max_n: int) -> Tensor:
    first = values[0]
    output = first.new_zeros((len(values), max_n, *first.shape[2:]))
    for index, value in enumerate(values):
        if value.shape[0] != 1 or value.device != first.device or value.dtype != first.dtype:
            raise ValueError("collate expects single-item tensors with matching device and dtype")
        output[index, : value.shape[1]] = value[0]
    return output


def _pad_time_entities(values: Sequence[Tensor], max_t: int, max_n: int) -> Tensor:
    first = values[0]
    output = first.new_zeros((len(values), max_t, max_n, *first.shape[3:]))
    for index, value in enumerate(values):
        if value.shape[0] != 1 or value.device != first.device or value.dtype != first.dtype:
            raise ValueError("collate expects single-item tensors with matching device and dtype")
        output[index, : value.shape[1], : value.shape[2]] = value[0]
    return output


def _pad_entity_pairs(values: Sequence[Tensor], max_n: int) -> Tensor:
    first = values[0]
    output = first.new_zeros((len(values), max_n, max_n))
    for index, value in enumerate(values):
        if value.shape[0] != 1 or value.device != first.device or value.dtype != first.dtype:
            raise ValueError("collate expects single-item tensors with matching device and dtype")
        n = value.shape[1]
        output[index, :n, :n] = value[0]
    return output


def collate_episodes(
    samples: Sequence[tuple[EvidenceBatch, TrainingTargets]],
) -> tuple[EvidenceBatch, TrainingTargets]:
    """Concatenate batch-one records, padding variable history and target-object counts."""
    if not samples:
        raise ValueError("collate_episodes requires at least one sample")
    evidence_rows, target_rows = zip(*samples)
    if any(row.images.shape[0] != 1 for row in evidence_rows):
        raise ValueError("each input sample must have leading batch dimension 1")
    if any(row.poses.shape[0] != 1 for row in target_rows):
        raise ValueError("each target sample must have leading batch dimension 1")
    max_t = max(row.images.shape[1] for row in evidence_rows)
    max_n = max(row.poses.shape[1] for row in target_rows)

    time_fields = (
        "images",
        "pixel_valid",
        "frame_valid",
        "camera_valid",
        "timestamps",
        "arrival_times",
        "actions",
        "robots",
        "calibration",
        "observed_results",
        "result_valid",
        "segment_ids",
        "anchor_poses",
        "source_ids",
    )
    evidence_values = {
        name: _pad_time([getattr(row, name) for row in evidence_rows], max_t)
        for name in time_fields
    }
    for name in ("query_time", "task", "metric_valid", "reference_version"):
        evidence_values[name] = torch.cat([getattr(row, name) for row in evidence_rows], dim=0)
    batched_evidence = EvidenceBatch(**evidence_values)

    target_time_entity = {"instance_masks", "visibility"}
    target_entity = {"poses", "pose_valid", "entity_valid"}
    target_pairs = {"change_times", "change_valid"}
    target_values: dict[str, Tensor | None] = {}
    for name in TARGET_FIELDS:
        rows = [getattr(row, name) for row in target_rows]
        if name in target_time_entity:
            target_values[name] = _pad_time_entities(rows, max_t, max_n)
        elif name in target_entity:
            target_values[name] = _pad_entities(rows, max_n)
        elif name in target_pairs:
            target_values[name] = _pad_entity_pairs(rows, max_n)
        elif name in {
            "future_image",
            "future_image_valid",
            "future_actions",
            "future_response",
            "future_response_valid",
            "future_success",
            "future_success_valid",
        }:
            target_values[name] = torch.cat(rows, dim=0)
        else:  # pragma: no cover - guarded by the schema's field list
            raise ValueError(f"unsupported target field: {name}")

    equivalent_rows = [row.equivalent_poses for row in target_rows]
    validity_rows = [row.equivalent_valid for row in target_rows]
    if any(
        (poses is None) != (valid is None) for poses, valid in zip(equivalent_rows, validity_rows)
    ):
        raise ValueError("equivalent target tensors must be supplied as a pair")
    if all(value is None for value in equivalent_rows + validity_rows):
        target_values["equivalent_poses"] = None
        target_values["equivalent_valid"] = None
    else:
        equivalent_rows = [
            row.poses[:, None] if value is None else value
            for row, value in zip(target_rows, equivalent_rows)
        ]
        validity_rows = [
            torch.ones(1, 1, dtype=torch.bool, device=row.poses.device) if value is None else value
            for row, value in zip(target_rows, validity_rows)
        ]
        max_l = max(value.shape[1] for value in equivalent_rows if value is not None)
        first = next(value for value in equivalent_rows if value is not None)
        eq = first.new_zeros((len(samples), max_l, max_n, 6))
        eq_valid = validity_rows[0].new_zeros((len(samples), max_l))  # type: ignore[union-attr]
        for index, (poses, valid) in enumerate(zip(equivalent_rows, validity_rows)):
            assert poses is not None and valid is not None
            if poses.shape[0] != 1 or poses.device != first.device or poses.dtype != first.dtype:
                raise ValueError(
                    "equivalent poses must have batch size one and matching device/dtype"
                )
            length, count = poses.shape[1:3]
            eq[index, :length, :count] = poses[0]
            eq_valid[index, :length] = valid[0]
        target_values["equivalent_poses"] = eq
        target_values["equivalent_valid"] = eq_valid

    point_rows = [row.image_points for row in target_rows]
    point_valid_rows = [row.image_points_valid for row in target_rows]
    if any(
        (points is None) != (valid is None) for points, valid in zip(point_rows, point_valid_rows)
    ):
        raise ValueError("image point tensors must be supplied as a pair")
    if all(value is None for value in point_rows + point_valid_rows):
        target_values["image_points"] = None
        target_values["image_points_valid"] = None
    else:
        point_rows = [
            row.poses.new_zeros(1, row.poses.shape[1], 2) if value is None else value
            for row, value in zip(target_rows, point_rows)
        ]
        point_valid_rows = [
            torch.zeros(1, row.poses.shape[1], dtype=torch.bool, device=row.poses.device)
            if value is None
            else value
            for row, value in zip(target_rows, point_valid_rows)
        ]
        first_points = next(value for value in point_rows if value is not None)
        points = first_points.new_zeros((len(samples), max_n, 2))
        points_valid = point_valid_rows[0].new_zeros((len(samples), max_n))  # type: ignore[union-attr]
        for index, (value, valid) in enumerate(zip(point_rows, point_valid_rows)):
            assert value is not None and valid is not None
            if (
                value.shape[0] != 1
                or value.device != first_points.device
                or value.dtype != first_points.dtype
            ):
                raise ValueError("image points must have batch size one and matching device/dtype")
            count = value.shape[1]
            points[index, :count] = value[0]
            points_valid[index, :count] = valid[0]
        target_values["image_points"] = points
        target_values["image_points_valid"] = points_valid

    for name, valid_name in (
        ("relation_relevance", "relation_relevance_valid"),
        ("contact_states", "contact_valid"),
    ):
        rows = [getattr(row, name) for row in target_rows]
        valid_rows = [getattr(row, valid_name) for row in target_rows]
        if any((value is None) != (valid is None) for value, valid in zip(rows, valid_rows)):
            raise ValueError(f"{name} and {valid_name} must be supplied as a pair")
        if all(value is None for value in rows + valid_rows):
            target_values[name] = None
            target_values[valid_name] = None
        else:
            dtype = next(value.dtype for value in rows if value is not None)
            rows = [
                torch.zeros(
                    1, row.poses.shape[1], row.poses.shape[1], dtype=dtype, device=row.poses.device
                )
                if value is None
                else value
                for row, value in zip(target_rows, rows)
            ]
            valid_rows = [
                torch.zeros(
                    1,
                    row.poses.shape[1],
                    row.poses.shape[1],
                    dtype=torch.bool,
                    device=row.poses.device,
                )
                if value is None
                else value
                for row, value in zip(target_rows, valid_rows)
            ]
            target_values[name] = _pad_entity_pairs(rows, max_n)  # type: ignore[arg-type]
            target_values[valid_name] = _pad_entity_pairs(valid_rows, max_n)  # type: ignore[arg-type]
    batched_targets = TrainingTargets(**target_values)  # type: ignore[arg-type]
    return batched_evidence, batched_targets


def _unbatch_dataclass(value: Any, allowed: Sequence[str]) -> dict[str, np.ndarray]:
    if not is_dataclass(value):
        return {name: _numpy(item) for name, item in value.items()}
    result: dict[str, np.ndarray] = {}
    for field in fields(value):
        name = field.name
        if name not in allowed:
            continue
        item = getattr(value, name)
        if item is None:
            continue
        array = _numpy(item)
        if array.ndim == 0 or array.shape[0] != 1:
            raise ValueError(f"export expects a batch-one {name} tensor")
        result[name] = array[0]
    return result


def export_recorder_npz(
    path: str | Path,
    sensor: Mapping[str, Any] | EvidenceBatch,
    targets: Mapping[str, Any] | TrainingTargets,
    *,
    config: Config,
    system_id: str,
    episode_id: str,
    split: str = "train",
) -> dict[str, Any]:
    """Export one validated, unbatched sensor/target pair as the recorder NPZ contract."""
    sensor_arrays = (
        _unbatch_dataclass(sensor, SENSOR_FIELDS)
        if is_dataclass(sensor)
        else {name: _numpy(value) for name, value in sensor.items()}
    )
    target_names = (*TARGET_FIELDS, *OPTIONAL_TARGET_FIELDS)
    target_arrays = (
        _unbatch_dataclass(targets, target_names)
        if is_dataclass(targets)
        else {name: _numpy(value) for name, value in targets.items()}
    )
    validate_episode_arrays(sensor_arrays, target_arrays, config)
    arrays = build_record_arrays(
        sensor_arrays,
        target_arrays,
        system_id=system_id,
        episode_id=episode_id,
        split=split,
        backend="recorder_npz",
    )
    digest = write_npz(path, arrays)
    return {"format_version": FORMAT_VERSION, "sha256": digest, "bytes": Path(path).stat().st_size}


def import_recorder_npz(
    source: str | Path,
    output_root: str | Path,
    *,
    config: Config,
    system_id: str | None = None,
    episode_id: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Import one exported/contract-compliant recording into a manifest-backed dataset."""
    source = Path(source)
    source_sensor, source_targets, source_meta = split_record_arrays(read_npz(source))
    if source_meta["backend"] != "recorder_npz":
        raise ValueError("input must use the recorder_npz export contract")
    resolved_system = system_id or source_meta["system_id"]
    resolved_episode = episode_id or source_meta["episode_id"]
    resolved_split = split or source_meta["split"]
    if (resolved_system, resolved_episode, resolved_split) != (
        source_meta["system_id"],
        source_meta["episode_id"],
        source_meta["split"],
    ):
        raise ValueError("import identity and split must match the NPZ source metadata")
    validate_episode_arrays(source_sensor, source_targets, config)

    root = Path(output_root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = validate_manifest(root, config)
        if manifest["backend"] != RECORDER_BACKEND:
            raise ValueError("cannot mix recorder imports with another dataset backend")
        if manifest["config"] != config.to_dict():
            raise ValueError("recorder import config differs from the existing dataset")
        system_splits = {key: list(value) for key, value in manifest["system_splits"].items()}
        episodes = list(manifest["episodes"])
    else:
        system_splits = {key: [] for key in SPLITS}
        episodes = []
    if any(
        item["system_id"] == resolved_system and item["episode_id"] == resolved_episode
        for item in episodes
    ):
        raise FileExistsError(f"episode already imported: {resolved_system}/{resolved_episode}")
    existing_split = next(
        (group for group in SPLITS if resolved_system in system_splits[group]), None
    )
    if existing_split is not None and existing_split != resolved_split:
        raise ValueError("a system cannot be assigned to more than one split")
    if existing_split is None:
        system_splits[resolved_split].append(resolved_system)
        system_splits[resolved_split].sort()

    relative = _episode_path(resolved_system, resolved_episode)
    destination = _record_path(root, relative)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing episode file: {relative}")
    arrays = build_record_arrays(
        source_sensor,
        source_targets,
        system_id=resolved_system,
        episode_id=resolved_episode,
        split=resolved_split,
        backend=RECORDER_BACKEND,
    )
    digest = write_npz(destination, arrays)
    episodes.append(
        {
            "path": relative,
            "split": resolved_split,
            "system_id": resolved_system,
            "episode_id": resolved_episode,
            "sha256": digest,
        }
    )
    episodes.sort(key=lambda item: (item["system_id"], item["episode_id"]))
    manifest = make_manifest(
        config, backend=RECORDER_BACKEND, system_splits=system_splits, episodes=episodes
    )
    write_manifest(root, manifest)
    return validate_manifest(root, config)
