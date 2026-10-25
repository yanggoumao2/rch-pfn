"""Import native LabUtopia insertion recorder JSON without discarding source fields."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ..config import Config
from .dataset import (
    LABUTOPIA_BACKEND,
    MANIFEST_VERSION,
    _episode_path,
    _record_path,
    make_manifest,
    validate_manifest,
    write_manifest,
)
from .format import (
    RECORDER_FIELDS,
    build_record_arrays,
    safe_identifier,
    stable_source_id,
    validate_episode_arrays,
    validate_recorder_arrays,
    write_npz,
)

PHASES = (
    "post_reset",
    "grasp_initialization",
    "hold",
    "descend_from_above",
    "insert",
    "seat",
    "release",
    "withdraw",
    "lift_clear",
    "settle",
)
_PHASE_IDS = {name: index for index, name in enumerate(PHASES)}


def _json_field_paths(value: Any, path: str = "") -> list[str]:
    """Enumerate every JSON leaf using a stable wildcard path for arrays."""
    if isinstance(value, dict):
        paths: list[str] = []
        for key in sorted(value):
            paths.extend(_json_field_paths(value[key], f"{path}/{key}"))
        return paths or [path or "/"]
    if isinstance(value, list):
        if not value:
            return [path or "/"]
        if all(not isinstance(item, (dict, list)) for item in value):
            return [path or "/"]
        if all(_is_numeric_tree(item) for item in value):
            return [path or "/"]
        paths: list[str] = []
        for item in value:
            paths.extend(_json_field_paths(item, f"{path}/*"))
        return sorted(set(paths))
    return [path or "/"]


def _is_numeric_tree(value: Any) -> bool:
    if isinstance(value, list):
        return all(_is_numeric_tree(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _field_disposition(path: str) -> str:
    path = path.replace("/left/", "/*/").replace("/right/", "/*/")
    if path in {
        "/frames/*/joint_positions",
        "/frames/*/joint_velocities",
        "/frames/*/joint_targets",
        "/frames/*/applied_targets",
    }:
        return "model_input"
    if path in {
        "/frames/*/positions",
        "/frames/*/orientations_wxyz",
        "/frames/*/grasp_contacts/*/force_world_n",
        "/frames/*/grasp_contacts/*/normal_force_sum_n",
        "/frames/*/grasp_contacts/*/points_world_m/*",
        "/frames/*/grasp_contacts/*/normals_world/*",
        "/frames/*/grasp_contacts/*/separations_m/*",
        "/frames/*/grasp_contacts/*/count",
        "/actual_pose_samples_480hz/*/hand_position_world_m/*",
        "/actual_pose_samples_480hz/*/hand_orientation_wxyz/*",
        "/actual_pose_samples_480hz/*/tube_position_world_m/*",
        "/actual_pose_samples_480hz/*/tube_orientation_wxyz/*",
    }:
        return "oracle_only_input"
    if path.startswith("/result/") or path.startswith("/success") or path.startswith("/experiment_success"):
        return "training_target_or_metadata"
    return "archive_only"


def _source_field_manifest(
    motion: dict[str, Any],
    outcome: dict[str, Any],
    policy: str,
    include_outcome: bool,
) -> list[dict[str, str]]:
    fields = [
        {"source": "motion.json", "path": path, "disposition": _field_disposition(path)}
        for path in _json_field_paths(motion)
    ]
    if include_outcome:
        fields.extend(
            {
                "source": "outcome.json",
                "path": path,
                "disposition": _field_disposition(path),
            }
            for path in _json_field_paths(outcome)
        )
    for item in fields:
        item["policy"] = policy
    return fields


def _json_bytes(path: Path) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid LabUtopia JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"LabUtopia JSON root must be an object: {path}")
    return payload, value


def _rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("LabUtopia recorder contains a zero quaternion")
    xyzw = (quaternion / norm)[..., [1, 2, 3, 0]]
    return Rotation.from_quat(xyzw).as_matrix()


def _phase_id(value: str) -> int:
    if value not in _PHASE_IDS:
        raise ValueError(
            f"unsupported LabUtopia phase {value!r}; add it to the fixed phase registry"
        )
    return _PHASE_IDS[value]


def _body_indices(body_paths: list[str], expected_count: int) -> tuple[int, int, int]:
    if len(body_paths) != expected_count:
        raise ValueError(f"motion.json body_paths must identify all {expected_count} recorded bodies")
    expected_suffixes = [f"/panda_link{index}" for index in range(9)] + [
        "/panda_hand",
        "/panda_leftfinger",
        "/panda_rightfinger",
        "/TubeReference",
    ]
    actual_suffixes = [
        next((suffix for suffix in expected_suffixes if path.endswith(suffix)), "")
        for path in body_paths
    ]
    if actual_suffixes != expected_suffixes:
        raise ValueError("body_paths must use the recorder's canonical 13-body identity order")
    return 0, 9, 12


def _select_frames(frames: list[dict[str, Any]], capacity: int) -> np.ndarray:
    if len(frames) <= capacity:
        return np.arange(len(frames), dtype=np.int64)
    phase_ids = [str(frame.get("phase", "")) for frame in frames]
    mandatory = {0, len(frames) - 1}
    mandatory.update(index for index in range(1, len(frames)) if phase_ids[index] != phase_ids[index - 1])
    if len(mandatory) > capacity:
        raise ValueError("configured max_frames cannot retain every recorder phase boundary")
    selected = set(mandatory)
    while len(selected) < capacity:
        ordered = sorted(selected)
        candidates = range(len(frames))
        candidate = max(
            (index for index in candidates if index not in selected),
            key=lambda index: min(abs(index - known) for known in ordered),
        )
        selected.add(candidate)
    return np.asarray(sorted(selected), dtype=np.int64)


def _frame_recorder(motion: dict[str, Any], config: Config, policy: str) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    frames = motion.get("frames")
    if not isinstance(frames, list) or not frames or not all(isinstance(row, dict) for row in frames):
        raise ValueError("motion.json must contain a nonempty frames array")
    indices = _select_frames(frames, config.model.max_frames)
    selected = [frames[int(index)] for index in indices]
    times = np.asarray([row["time"] for row in selected], dtype=np.float64)
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError("motion frame times must be finite and chronological")

    positions_world = np.asarray([row["positions"] for row in selected], dtype=np.float64)
    orientations_world = np.asarray([row["orientations_wxyz"] for row in selected], dtype=np.float64)
    joints = np.asarray([row["joint_positions"] for row in selected], dtype=np.float64)
    velocities = np.asarray([row["joint_velocities"] for row in selected], dtype=np.float64)
    joint_targets = np.asarray([row["joint_targets"] for row in selected], dtype=np.float64)
    applied_targets = np.asarray([row["applied_targets"] for row in selected], dtype=np.float64)
    body_count, joint_count = config.model.recorder_body_count, config.model.recorder_joint_count
    body_paths = [str(value) for value in motion.get("body_paths", [])]
    base_index, _, _ = _body_indices(body_paths, body_count)
    if positions_world.shape != (len(selected), body_count, 3):
        raise ValueError(f"expected {body_count} body positions per frame, got {positions_world.shape}")
    if orientations_world.shape != (len(selected), body_count, 4):
        raise ValueError(f"expected {body_count} body quaternions per frame, got {orientations_world.shape}")
    if any(value.shape != (len(selected), joint_count) for value in (joints, velocities, joint_targets, applied_targets)):
        raise ValueError(f"expected {joint_count} joints in every motion field")
    if not all(np.isfinite(value).all() for value in (positions_world, orientations_world, joints, velocities, joint_targets, applied_targets)):
        raise ValueError("motion frames contain non-finite values")

    rotations_world = _rotation(orientations_world)
    base_position = positions_world[0, base_index]
    base_rotation = rotations_world[0, base_index]
    base_inverse = base_rotation.T
    positions_base = np.einsum("ij,tkj->tki", base_inverse, positions_world - base_position)
    rotations_base = np.einsum("ij,tkjl->tkil", base_inverse, rotations_world)
    orientations_base = Rotation.from_matrix(rotations_base.reshape(-1, 3, 3)).as_quat().reshape(
        len(selected), body_count, 4
    )[:, :, [3, 0, 1, 2]]
    phases = [str(row.get("phase", "")) for row in selected]
    if any(not phase for phase in phases):
        raise ValueError("every motion frame must include a phase")
    phase_ids_by_name = {phase: _phase_id(phase) for phase in set(phases)}
    if len(phase_ids_by_name) != len(set(phase_ids_by_name.values())):
        raise ValueError("phase hash collision; add the phase name to the fixed phase registry")
    local_phase_ids = [phase_ids_by_name[phase] for phase in phases]

    point_capacity = config.model.recorder_contact_points
    contact_force = np.zeros((len(selected), 2, 3), dtype=np.float32)
    normal_force = np.zeros((len(selected), 2), dtype=np.float32)
    contact_count = np.zeros((len(selected), 2), dtype=np.int64)
    contact_valid = np.zeros((len(selected), 2), dtype=bool)
    points = np.zeros((len(selected), 2, point_capacity, 3), dtype=np.float32)
    normals = np.zeros_like(points)
    separations = np.zeros((len(selected), 2, point_capacity), dtype=np.float32)
    point_valid = np.zeros((len(selected), 2, point_capacity), dtype=bool)
    side_names = ("left", "right")
    for ti, row in enumerate(selected):
        contacts = row.get("grasp_contacts")
        if not isinstance(contacts, dict):
            continue
        for side_index, side_name in enumerate(side_names):
            side = contacts.get(side_name)
            if not isinstance(side, dict):
                continue
            force = np.asarray(side.get("force_world_n", [0, 0, 0]), dtype=np.float64)
            side_points = np.asarray(side.get("points_world_m", []), dtype=np.float64).reshape(-1, 3)
            side_normals = np.asarray(side.get("normals_world", []), dtype=np.float64).reshape(-1, 3)
            side_separations = np.asarray(side.get("separations_m", []), dtype=np.float64).reshape(-1)
            count = int(side.get("count", len(side_points)))
            if force.shape != (3,) or count < 0 or count > point_capacity:
                raise ValueError(f"invalid {side_name} contact force/count at frame {ti}")
            if not (len(side_points) == len(side_normals) == len(side_separations) == count):
                raise ValueError(f"{side_name} contact point arrays disagree with count at frame {ti}")
            if not all(np.isfinite(value).all() for value in (force, side_points, side_normals, side_separations)):
                raise ValueError(f"non-finite {side_name} contact data at frame {ti}")
            contact_force[ti, side_index] = base_inverse @ force
            normal_force[ti, side_index] = float(side.get("normal_force_sum_n", 0.0))
            contact_count[ti, side_index] = count
            contact_valid[ti, side_index] = True
            points[ti, side_index, :count] = (base_inverse @ (side_points - base_position).T).T
            normals[ti, side_index, :count] = (base_inverse @ side_normals.T).T
            separations[ti, side_index, :count] = side_separations
            point_valid[ti, side_index, :count] = True

    trace_rows = motion.get("actual_pose_samples_480hz", [])
    trace_arrays: dict[str, np.ndarray] = {}
    if trace_rows:
        if not isinstance(trace_rows, list) or not all(isinstance(row, dict) for row in trace_rows):
            raise ValueError("actual_pose_samples_480hz must be an array of objects")
        trace_time_all = np.asarray([row["time_s"] for row in trace_rows], dtype=np.float64)
        trace_position_world = np.asarray(
            [[row["hand_position_world_m"], row["tube_position_world_m"]] for row in trace_rows],
            dtype=np.float64,
        )
        trace_orientation_world = np.asarray(
            [[row["hand_orientation_wxyz"], row["tube_orientation_wxyz"]] for row in trace_rows],
            dtype=np.float64,
        )
        if not np.isfinite(trace_time_all).all() or np.any(np.diff(trace_time_all) < 0):
            raise ValueError("480 Hz trace times must be finite and chronological")
        trace_indices = np.linspace(
            0, len(trace_rows) - 1,
            min(len(trace_rows), config.model.recorder_trace_frames),
            dtype=np.int64,
        )
        trace_time = trace_time_all[trace_indices]
        trace_rot_world = _rotation(trace_orientation_world[trace_indices])
        trace_position = np.einsum(
            "ij,hkj->hki", base_inverse, trace_position_world[trace_indices] - base_position
        )
        trace_rot = np.einsum("ij,hkjl->hkil", base_inverse, trace_rot_world)
        trace_orientation = Rotation.from_matrix(trace_rot.reshape(-1, 3, 3)).as_quat().reshape(
            len(trace_indices), 2, 4
        )[:, :, [3, 0, 1, 2]]
        trace_arrays = {
            "trace_times": trace_time.astype(np.float32),
            "trace_body_positions": trace_position.astype(np.float32),
            "trace_body_orientations_wxyz": trace_orientation.astype(np.float32),
            "trace_valid": np.ones((len(trace_indices),), dtype=bool),
        }

    recorder = {
        "body_positions": positions_base.astype(np.float32),
        "body_orientations_wxyz": orientations_base.astype(np.float32),
        "body_valid": np.ones((len(selected), body_count), dtype=bool),
        "joint_positions": joints.astype(np.float32),
        "joint_velocities": velocities.astype(np.float32),
        "joint_targets": joint_targets.astype(np.float32),
        "applied_targets": applied_targets.astype(np.float32),
        "joint_valid": np.ones((len(selected), joint_count), dtype=bool),
        "contact_force_base": contact_force,
        "contact_normal_force": normal_force,
        "contact_count": contact_count,
        "contact_valid": contact_valid,
        "contact_points_base": points,
        "contact_normals_base": normals,
        "contact_separations": separations,
        "contact_point_valid": point_valid,
        "phase_ids": np.asarray(local_phase_ids, dtype=np.int64),
        "phase_valid": np.ones((len(selected),), dtype=bool),
        "sample_time": times.astype(np.float32),
        "sample_valid": np.ones((len(selected),), dtype=bool),
        **trace_arrays,
    }
    if policy == "deployable":
        # Body transforms and physics-solver contacts are simulator state. Keep them
        # in raw archives and targets, but do not expose them as deployable inputs.
        recorder["body_valid"][:, :] = False
        recorder["contact_valid"][:, :] = False
        recorder["contact_point_valid"][:, :] = False
        recorder["phase_valid"][:] = False
        if "trace_valid" in recorder:
            recorder["trace_valid"][:] = False
    return recorder, indices, body_paths


def _source_archives(motion_path: Path, motion_bytes: bytes, outcome_path: Path | None, outcome_bytes: bytes | None, field_manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    archives: dict[str, np.ndarray] = {
        "motion_json": np.frombuffer(motion_bytes, dtype=np.uint8).copy(),
        "field_manifest_json": np.asarray(json.dumps(field_manifest, sort_keys=True, ensure_ascii=False)),
    }
    if outcome_bytes is not None:
        archives["outcome_json"] = np.frombuffer(outcome_bytes, dtype=np.uint8).copy()
    source_manifest = []
    source_candidates = sorted(motion_path.parent.glob("*.json"))
    for path in source_candidates:
        if path == motion_path or (outcome_path is not None and path == outcome_path):
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        key = f"source_file_{digest}"
        archives[key] = np.frombuffer(payload, dtype=np.uint8).copy()
        source_manifest.append(
            {"name": path.name, "array_key": key, "sha256": digest, "bytes": len(payload)}
        )
    archive_manifest = {
        "artifacts": [
            {"name": motion_path.name, "array_key": "motion_json", "sha256": hashlib.sha256(motion_bytes).hexdigest(), "bytes": len(motion_bytes)},
            *(
                [{"name": outcome_path.name, "array_key": "outcome_json", "sha256": hashlib.sha256(outcome_bytes).hexdigest(), "bytes": len(outcome_bytes)}]
                if outcome_path is not None and outcome_bytes is not None else []
            ),
            *source_manifest,
        ]
    }
    archives["source_manifest_json"] = np.asarray(
        json.dumps(archive_manifest, sort_keys=True, ensure_ascii=False)
    )
    return archives


def import_labutopia_episode(
    motion_path: str | Path,
    output_root: str | Path,
    *,
    config: Config,
    system_id: str | None = None,
    episode_id: str | None = None,
    split: str = "train",
    policy: str = "deployable",
) -> dict[str, Any]:
    """Convert a LabUtopia motion/outcome pair into a hashed recorder-enabled dataset."""
    motion_path = Path(motion_path).resolve()
    if motion_path.name not in {"motion.json", "partial-motion.json"}:
        raise ValueError("motion_path must point to motion.json or partial-motion.json")
    motion_bytes, motion = _json_bytes(motion_path)
    outcome_path = motion_path.with_name("outcome.json")
    outcome_bytes = None
    outcome: dict[str, Any] = motion.get("result", {})
    if outcome_path.exists():
        outcome_bytes, outcome = _json_bytes(outcome_path)
    if not isinstance(outcome, dict):
        outcome = {}
    if policy not in {"deployable", "recorder_oracle"}:
        raise ValueError("policy must be deployable or recorder_oracle")
    effective_config = replace(
        config,
        model=replace(config.model, recorder_enabled=True, recorder_policy=policy),
    )
    effective_config.validate()
    recorder, frame_indices, body_paths = _frame_recorder(motion, effective_config, policy)
    base_index, hand_index, tube_index = _body_indices(
        body_paths, effective_config.model.recorder_body_count
    )
    frames = motion["frames"]
    selected = [frames[int(index)] for index in frame_indices]
    times = np.asarray([row["time"] for row in selected], dtype=np.float32)
    count = len(selected)
    cfg = effective_config.model
    if cfg.max_cameras < 1:
        raise ValueError("at least one configured camera slot is required")
    camera_count, size = 1, cfg.image_size
    images = np.zeros((count, camera_count, 3, size, size), dtype=np.uint8)
    joint_positions = recorder["joint_positions"]
    joint_velocities = recorder["joint_velocities"]
    applied_targets = recorder["applied_targets"]
    sensor = {
        "images": images,
        "pixel_valid": np.zeros((count, camera_count, size, size), dtype=bool),
        "frame_valid": np.ones((count,), dtype=bool),
        "camera_valid": np.zeros((count, camera_count), dtype=bool),
        "timestamps": times,
        "arrival_times": times.copy(),
        "query_time": np.asarray(times[-1], dtype=np.float32),
        "actions": np.zeros((count, cfg.action_dim), dtype=np.float32),
        "robots": np.zeros((count, cfg.robot_dim), dtype=np.float32),
        "calibration": np.zeros((count, camera_count, 25), dtype=np.float32),
        "task": np.zeros((cfg.task_dim,), dtype=np.float32),
        "observed_results": np.zeros((count, cfg.outcome_dim), dtype=np.float32),
        "result_valid": np.zeros((count, cfg.outcome_dim), dtype=bool),
        "segment_ids": np.minimum(np.arange(count) * cfg.max_segments // count, cfg.max_segments - 1).astype(np.int64),
        "anchor_poses": np.repeat(np.eye(4, dtype=np.float32)[None, None], count, axis=0).repeat(2, axis=1),
        "metric_valid": np.asarray(True, dtype=bool),
        "source_ids": np.asarray(
            [stable_source_id(system_id or motion_path.parent.name, episode_id or motion_path.parent.name, int(index)) for index in frame_indices],
            dtype=np.int64,
        ),
        "reference_version": np.asarray(0, dtype=np.int64),
    }
    action_width = min(cfg.action_dim, cfg.recorder_joint_count)
    sensor["actions"][:, :action_width] = applied_targets[:, :action_width]
    if cfg.robot_dim >= 2 * cfg.recorder_joint_count:
        sensor["robots"][:, : cfg.recorder_joint_count] = joint_positions
        sensor["robots"][:, cfg.recorder_joint_count : 2 * cfg.recorder_joint_count] = joint_velocities
    else:
        sensor["robots"][:, :] = np.concatenate([joint_positions, joint_velocities], axis=-1)[:, : cfg.robot_dim]
    if policy == "recorder_oracle":
        base_orientations = recorder["body_orientations_wxyz"]
        base_positions = recorder["body_positions"]
        for index in range(count):
            for anchor_index, body_index in ((0, base_index), (1, hand_index)):
                quaternion = base_orientations[index, body_index]
                rotation = _rotation(quaternion)
                sensor["anchor_poses"][index, anchor_index, :3, :3] = rotation
                sensor["anchor_poses"][index, anchor_index, :3, 3] = base_positions[index, body_index]

    final_pose = recorder["body_positions"][-1, tube_index]
    final_rotation = _rotation(recorder["body_orientations_wxyz"][-1, tube_index])
    final_rotvec = Rotation.from_matrix(final_rotation).as_rotvec().astype(np.float32)
    success_value = outcome.get("experiment_success", outcome.get("task_success", outcome.get("success")))
    success_valid = isinstance(success_value, bool)
    targets = {
        "poses": np.concatenate([final_pose.astype(np.float32), final_rotvec])[None],
        "pose_valid": np.ones((1, 6), dtype=bool),
        "instance_masks": np.zeros((count, 1, size, size), dtype=bool),
        "entity_valid": np.ones((1,), dtype=bool),
        "visibility": np.zeros((count, 1), dtype=np.float32),
        "change_times": np.zeros((1, 1), dtype=np.int64),
        "change_valid": np.zeros((1, 1), dtype=bool),
        "future_actions": np.zeros((cfg.action_dim,), dtype=np.float32),
        "future_response": np.zeros((cfg.robot_dim,), dtype=np.float32),
        "future_response_valid": np.zeros((cfg.robot_dim,), dtype=bool),
        "future_image": np.zeros((camera_count, 3, size, size), dtype=np.uint8),
        "future_image_valid": np.zeros((camera_count,), dtype=bool),
        "future_success": np.asarray(float(bool(success_value)) if success_valid else 0.0, dtype=np.float32),
        "future_success_valid": np.asarray(success_valid, dtype=bool),
    }
    field_manifest = {
        "schema": "labutopia-insertion-recorder-v1",
        "policy": policy,
        "body_paths": body_paths,
        "normalized_coordinate_frame": "initial_robot_base",
        "joint_order": [f"panda_joint{i}" for i in range(1, 8)] + ["panda_finger_joint1", "panda_finger_joint2"],
        "contact_pairs": ["left_finger->tube", "right_finger->tube"],
        "selected_frame_indices": frame_indices.tolist(),
        "raw_frames": len(frames),
        "model_frames": count,
        "raw_trace_samples": len(motion.get("actual_pose_samples_480hz", [])),
        "model_trace_samples": len(recorder.get("trace_times", [])),
        "all_source_fields": _source_field_manifest(
            motion, outcome, policy, outcome_bytes is not None
        ),
        "all_fields_preserved": True,
        "field_disposition": {
            "model_input": ["frames[].joint_positions", "frames[].joint_velocities", "frames[].joint_targets", "frames[].applied_targets"],
            "oracle_only_input": ["frames[].positions", "frames[].orientations_wxyz", "frames[].grasp_contacts", "actual_pose_samples_480hz[]"],
            "training_target": ["final tube pose", "outcome experiment_success/task_success/success"],
            "archive_only": ["all unprojected/unknown JSON leaves, 480 Hz source trace, geometry audits, run configuration, source hashes"],
            "raw_source_preserved": True,
        },
    }
    archives = _source_archives(motion_path, motion_bytes, outcome_path if outcome_bytes is not None else None, outcome_bytes, field_manifest)
    validate_episode_arrays(sensor, targets, effective_config, recorder=recorder)
    validate_recorder_arrays(recorder, effective_config)

    resolved_system = safe_identifier(system_id or motion_path.parent.name, "system_id")
    resolved_episode = safe_identifier(episode_id or motion_path.parent.name, "episode_id")
    root = Path(output_root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = validate_manifest(root, effective_config)
        if manifest["backend"] != LABUTOPIA_BACKEND:
            raise ValueError("cannot mix LabUtopia records with another dataset backend")
        if manifest["config"] != effective_config.to_dict():
            raise ValueError("LabUtopia import config differs from the existing dataset")
        groups = {key: list(value) for key, value in manifest["system_splits"].items()}
        episodes = list(manifest["episodes"])
    else:
        groups = {key: [] for key in ("train", "val", "test")}
        episodes = []
    if any(item["system_id"] == resolved_system and item["episode_id"] == resolved_episode for item in episodes):
        raise FileExistsError(f"episode already imported: {resolved_system}/{resolved_episode}")
    assigned = next((name for name, values in groups.items() if resolved_system in values), None)
    if assigned is not None and assigned != split:
        raise ValueError("a system cannot be assigned to more than one split")
    if assigned is None:
        groups[split].append(resolved_system)
        groups[split].sort()
    relative = _episode_path(resolved_system, resolved_episode)
    destination = _record_path(root, relative)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing episode file: {relative}")
    arrays = build_record_arrays(
        sensor,
        targets,
        system_id=resolved_system,
        episode_id=resolved_episode,
        split=split,
        backend=LABUTOPIA_BACKEND,
        recorder=recorder,
        archives=archives,
    )
    digest = write_npz(destination, arrays)
    episodes.append({"path": relative, "split": split, "system_id": resolved_system, "episode_id": resolved_episode, "sha256": digest})
    episodes.sort(key=lambda item: (item["system_id"], item["episode_id"]))
    manifest = make_manifest(effective_config, backend=LABUTOPIA_BACKEND, system_splits=groups, episodes=episodes)
    manifest["schema_version"] = MANIFEST_VERSION
    write_manifest(root, manifest)
    validated = validate_manifest(root, effective_config)
    return {
        "backend": LABUTOPIA_BACKEND,
        "policy": policy,
        "episode": relative,
        "raw_motion_sha256": hashlib.sha256(motion_bytes).hexdigest(),
        "raw_outcome_sha256": hashlib.sha256(outcome_bytes).hexdigest() if outcome_bytes is not None else None,
        "raw_frame_count": len(frames),
        "model_frame_count": count,
        "raw_trace_count": len(motion.get("actual_pose_samples_480hz", [])),
        "model_trace_count": len(recorder.get("trace_times", [])),
        "recorder_fields": [f"recorder__{name}" for name in RECORDER_FIELDS if name in recorder],
        "archive_fields": [f"archive__{name}" for name in archives],
        "manifest_episodes": len(validated["episodes"]),
    }
