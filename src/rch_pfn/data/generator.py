"""Deterministic procedural RGB kinematic data (not a contact-physics simulator)."""

from __future__ import annotations

import colorsys
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ..config import Config
from .dataset import (
    SYNTHETIC_BACKEND,
    make_manifest,
    validate_manifest,
    write_manifest,
)
from .format import (
    build_record_arrays,
    safe_identifier,
    stable_seed,
    stable_source_id,
    validate_episode_arrays,
    write_npz,
)


@dataclass(frozen=True)
class _Mechanism:
    axis: tuple[float, float]
    coupling_gain: float
    phase: float
    drift_rate: float


def _validate_generator_config(config: Config) -> None:
    config.validate()
    p = config.prior
    numeric = (
        p.dt,
        p.position_range,
        p.rotation_range,
        p.noise_std,
        p.change_probability,
        p.occlusion_probability,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("prior parameters must be finite")
    if p.dt <= 0 or p.position_range <= 0 or p.rotation_range < 0 or p.noise_std < 0:
        raise ValueError("prior time, workspace, rotation, and noise ranges are invalid")
    if not 0 <= p.change_probability <= 1 or not 0 <= p.occlusion_probability <= 1:
        raise ValueError("prior probabilities must be in [0,1]")


def _mechanism(seed: int) -> _Mechanism:
    rng = np.random.default_rng(seed)
    angle = float(rng.uniform(-math.pi, math.pi))
    return _Mechanism(
        axis=(math.cos(angle), math.sin(angle)),
        coupling_gain=float(rng.uniform(0.35, 0.85)),
        phase=float(rng.uniform(0, 2 * math.pi)),
        drift_rate=float(rng.uniform(0.55, 1.2)),
    )


def _background(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.linspace(-1.0, 1.0, size, dtype=np.float32)[:, None, None]
    base = np.asarray([190.0, 185.0, 170.0], dtype=np.float32)[None, None, :]
    grain = rng.normal(0, 1.7, size=(size, size, 1)).astype(np.float32)
    wash = 3.0 * y
    return np.clip(base + grain + wash, 0, 255).astype(np.uint8)


def _rotation_matrix_yaw(yaw: float) -> np.ndarray:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cosine, -sine, 0.0, 0.0],
            [sine, cosine, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _shape_points(
    cx: float, cy: float, radius: float, yaw: float, shape: str
) -> list[tuple[float, float]]:
    if shape == "circle":
        return []
    if shape == "square":
        local = ((-1, -1), (1, -1), (1, 1), (-1, 1))
    else:
        local = ((0, -1.15), (1, 0.75), (-1, 0.75))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return [
        (cx + radius * (x * cosine - y * sine), cy + radius * (x * sine + y * cosine))
        for x, y in local
    ]


def _project_world_point(
    point: np.ndarray, intrinsics: np.ndarray, camera_to_base: np.ndarray
) -> tuple[float, float, float]:
    base_to_camera = np.linalg.inv(camera_to_base)
    point_camera = base_to_camera[:3, :3] @ point[:3] + base_to_camera[:3, 3]
    depth = float(point_camera[2])
    if depth <= 1e-8:
        return math.nan, math.nan, depth
    pixel = intrinsics @ point_camera
    return float(pixel[0] / pixel[2]), float(pixel[1] / pixel[2]), depth


def _render(
    poses: np.ndarray,
    *,
    colors: list[tuple[int, int, int]],
    shapes: list[str],
    radii: np.ndarray,
    background: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_base: np.ndarray,
    noise_std: float,
    noise_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = int(background.shape[0])
    rgb = Image.fromarray(background.copy(), mode="RGB")
    draw = ImageDraw.Draw(rgb)
    labels = Image.new("I", (size, size), 0)
    label_draw = ImageDraw.Draw(labels)
    projected: list[tuple[float, float, float, int]] = []
    for index, pose in enumerate(poses):
        u, v, depth = _project_world_point(pose, intrinsics, camera_to_base)
        projected.append((u, v, depth, index))
    # The overhead camera sees elevated objects in front; later layers occlude earlier ones.
    for u, v, depth, index in sorted(projected, key=lambda item: item[2], reverse=True):
        if not math.isfinite(u) or depth <= 0:
            continue
        radius_px = max(1.5, float(intrinsics[0, 0] * radii[index] / depth))
        yaw = float(poses[index, 5])
        shape = shapes[index]
        color = colors[index]
        dark = tuple(max(0, int(channel * 0.48)) for channel in color)
        if shape == "circle":
            box = (u - radius_px, v - radius_px, u + radius_px, v + radius_px)
            draw.ellipse(box, fill=color, outline=dark, width=max(1, size // 180))
            label_draw.ellipse(box, fill=index + 1)
            dot_u = u + math.cos(yaw) * radius_px * 0.3
            dot_v = v + math.sin(yaw) * radius_px * 0.3
            dot_r = max(1.0, radius_px * 0.12)
            draw.ellipse((dot_u - dot_r, dot_v - dot_r, dot_u + dot_r, dot_v + dot_r), fill=dark)
        else:
            points = _shape_points(u, v, radius_px, yaw, shape)
            draw.polygon(points, fill=color, outline=dark)
            label_draw.polygon(points, fill=index + 1)
            mark = (u + math.cos(yaw) * radius_px * 0.28, v + math.sin(yaw) * radius_px * 0.28)
            mark_r = max(1.0, radius_px * 0.1)
            draw.ellipse(
                (mark[0] - mark_r, mark[1] - mark_r, mark[0] + mark_r, mark[1] + mark_r), fill=dark
            )

    label_array = np.asarray(labels, dtype=np.int32)
    masks = np.stack([label_array == index + 1 for index in range(len(poses))], axis=0)
    visibility = np.zeros((len(poses),), dtype=np.float32)
    for index, pose in enumerate(poses):
        u, v, depth = _project_world_point(pose, intrinsics, camera_to_base)
        if not math.isfinite(u) or depth <= 0:
            continue
        radius_px = max(1.5, float(intrinsics[0, 0] * radii[index] / depth))
        mask = Image.new("L", (size, size), 0)
        mask_draw = ImageDraw.Draw(mask)
        if shapes[index] == "circle":
            mask_draw.ellipse(
                (u - radius_px, v - radius_px, u + radius_px, v + radius_px), fill=255
            )
        else:
            mask_draw.polygon(
                _shape_points(u, v, radius_px, float(pose[5]), shapes[index]), fill=255
            )
        total = int(np.asarray(mask, dtype=np.uint8).sum() // 255)
        visible = int(masks[index].sum())
        visibility[index] = visible / total if total else 0.0

    image = np.asarray(rgb, dtype=np.uint8)
    if noise_std:
        rng = np.random.default_rng(noise_seed)
        noise = rng.normal(0.0, noise_std * 255.0, size=image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, masks, visibility


def _make_task(task_dim: int, target_color: tuple[int, int, int], goal: np.ndarray) -> np.ndarray:
    task = np.zeros(task_dim, dtype=np.float32)
    for index, value in enumerate(target_color[:3]):
        if index < task_dim:
            task[index] = value / 255.0
    for index, value in enumerate(goal[:3], start=3):
        if index < task_dim:
            task[index] = value
    if task_dim > 6:
        task[6] = 1.0  # move-to-goal instruction family
    return task


def _command_sequence(rng: np.random.Generator, config: Config) -> np.ndarray:
    p, dims = config.prior, config.model.action_dim
    commands = np.zeros((p.frames, dims), dtype=np.float32)
    for step in range(p.frames):
        translation = rng.uniform(-0.12, 0.12, size=3) * p.position_range
        rotation = np.asarray([0.0, 0.0, rng.uniform(-0.12, 0.12) * p.rotation_range])
        commands[step, : min(3, dims)] = translation[: min(3, dims)]
        if dims > 3:
            count = min(3, dims - 3)
            commands[step, 3 : 3 + count] = rotation[:count]
        if dims > 6:
            commands[step, 6] = float(rng.random() > 0.5)
        if dims > 7:
            commands[step, 7] = 1.0
    return commands


def _decode_command(command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.zeros(3, dtype=np.float32)
    rotation = np.zeros(3, dtype=np.float32)
    translation[: min(3, command.size)] = command[: min(3, command.size)]
    if command.size > 3:
        count = min(3, command.size - 3)
        rotation[:count] = command[3 : 3 + count]
    return translation, rotation


def _robot_features(
    states: np.ndarray,
    frame_index: int,
    actor: int,
    commands: np.ndarray,
    config: Config,
    gripper_height: float,
) -> np.ndarray:
    """Synthetic gripper pose, aperture, and twist at one trajectory state."""
    features = np.zeros((13,), dtype=np.float32)
    features[:6] = states[frame_index, actor]
    features[2] += max(gripper_height, 0.02 * config.prior.position_range)
    if config.model.action_dim > 6 and frame_index > 0:
        features[6] = commands[frame_index - 1, 6]
    if frame_index > 0:
        velocity = (states[frame_index, actor] - states[frame_index - 1, actor]) / config.prior.dt
        features[7:13] = velocity
    return features


def _simulate_states(
    rng: np.random.Generator,
    config: Config,
    mechanism: _Mechanism,
    object_count: int,
    actor: int,
    follower: int | None,
    activation_step: int | None,
    radii: np.ndarray,
    elevations: np.ndarray,
    occlude_pair: bool,
) -> tuple[np.ndarray, np.ndarray]:
    p = config.prior
    commands = _command_sequence(rng, config)
    states = np.zeros((p.frames + 1, object_count, 6), dtype=np.float32)
    states[0, :, :2] = rng.uniform(-0.68, 0.68, size=(object_count, 2)) * p.position_range
    states[0, :, 2] = radii + elevations
    states[0, :, 5] = rng.uniform(-p.rotation_range, p.rotation_range, size=object_count)
    if occlude_pair and follower is not None:
        states[0, follower, :2] = states[0, actor, :2] + np.asarray(
            [0.65 * (radii[actor] + radii[follower]), 0], dtype=np.float32
        )

    axis = np.asarray(mechanism.axis, dtype=np.float32)
    perpendicular = np.asarray([-axis[1], axis[0]], dtype=np.float32)
    phase = rng.uniform(0, 2 * math.pi, size=object_count)
    for step in range(p.frames):
        states[step + 1] = states[step]
        command_delta, command_rotation = _decode_command(commands[step])
        passive = axis * (
            0.012 * p.position_range * math.sin(mechanism.phase + step * 0.7)
        ) + perpendicular * (0.008 * p.position_range * math.cos(mechanism.phase + step))
        actor_delta = command_delta.copy()
        actor_delta[:2] += passive
        states[step + 1, actor, :3] = states[step, actor, :3]
        states[step + 1, actor, :2] += actor_delta[:2]
        states[step + 1, actor, 5] += command_rotation[2]

        relation_active = activation_step is not None and step >= activation_step
        for entity in range(object_count):
            if entity == actor:
                continue
            angle = phase[entity] + mechanism.drift_rate * step
            ambient = axis * (0.018 * p.position_range * math.sin(angle)) + perpendicular * (
                0.013 * p.position_range * math.cos(angle)
            )
            states[step + 1, entity, :2] += ambient
            if relation_active and entity == follower:
                states[step + 1, entity, :2] += mechanism.coupling_gain * actor_delta[:2]
                states[step + 1, entity, 5] += mechanism.coupling_gain * command_rotation[2]

        bound = 0.92 * p.position_range
        states[step + 1, :, 0:2] = np.clip(states[step + 1, :, 0:2], -bound, bound)
        states[step + 1, :, 2] = radii + elevations
        states[step + 1, :, 3:5] = 0.0
        states[step + 1, :, 5] = np.clip(
            states[step + 1, :, 5], -p.rotation_range, p.rotation_range
        )
    return states, commands


def _episode_data(
    config: Config,
    *,
    system_id: str,
    episode_id: str,
    split: str,
    episode_seed: int,
    mechanism: _Mechanism,
    background: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    p, m = config.prior, config.model
    rng = np.random.default_rng(episode_seed)
    count = int(rng.integers(p.min_objects, p.max_objects + 1))
    actor = int(rng.integers(count))
    follower = int(rng.integers(count - 1)) if count > 1 else None
    if follower is not None and follower >= actor:
        follower += 1

    event_sampled = follower is not None and rng.random() < p.change_probability
    if event_sampled:
        # Step zero denotes a relation that was already active before the recording window.
        activation_step = 0 if rng.random() < 0.12 else int(rng.integers(1, p.frames + 1))
    else:
        activation_step = None

    radii = rng.uniform(0.08, 0.13, size=count).astype(np.float32) * p.position_range
    elevations = rng.uniform(0.0, 0.05, size=count).astype(np.float32) * p.position_range
    shapes = [str(rng.choice(("circle", "square", "triangle"))) for _ in range(count)]
    colors: list[tuple[int, int, int]] = []
    for _ in range(count):
        hue = float(rng.random())
        saturation = float(rng.uniform(0.55, 0.88))
        value = float(rng.uniform(0.62, 0.95))
        colors.append(
            tuple(
                int(round(channel * 255)) for channel in colorsys.hsv_to_rgb(hue, saturation, value)
            )
        )
    goal = rng.uniform(-0.75, 0.75, size=3).astype(np.float32) * p.position_range
    goal[2] = radii[actor] + elevations[actor]
    occlude_pair = follower is not None and rng.random() < p.occlusion_probability

    # Simulate the complete trajectory first; only then choose the causal observation prefix.
    states, commands = _simulate_states(
        rng,
        config,
        mechanism,
        count,
        actor,
        follower,
        activation_step,
        radii,
        elevations,
        bool(occlude_pair),
    )
    last_capture = int(rng.integers(1, p.frames))
    capture_times = np.arange(last_capture + 1, dtype=np.float64) * p.dt
    delays = rng.uniform(0, 1.5 * p.dt, size=last_capture + 1)
    delays[0] = 0.0
    delays[last_capture] = 0.0  # query is the latest exposure time; future target is strictly later
    query_time = float(capture_times[last_capture] + delays[last_capture])
    arrivals = capture_times + delays
    observed_indices = [index for index in range(last_capture + 1) if arrivals[index] <= query_time]
    if len(observed_indices) < 2:
        raise RuntimeError("causal prefix sampler must retain the first and latest capture")

    size = m.image_size
    camera_height = 0.75
    focal = size * 0.80 / (2.0 * p.position_range) * camera_height
    intrinsics = np.asarray(
        [
            [focal, 0.0, (size - 1) / 2.0],
            [0.0, focal, (size - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_to_base = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, camera_height],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    calibration = np.concatenate((intrinsics.reshape(-1), camera_to_base.reshape(-1)))

    rendered: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for frame_index in (*observed_indices, last_capture + 1):
        rendered[frame_index] = _render(
            states[frame_index],
            colors=colors,
            shapes=shapes,
            radii=radii,
            background=background,
            intrinsics=intrinsics,
            camera_to_base=camera_to_base,
            noise_std=p.noise_std,
            noise_seed=stable_seed(episode_seed, "rgb-noise", frame_index),
        )

    observed_count = len(observed_indices)
    images = np.stack([rendered[index][0].transpose(2, 0, 1) for index in observed_indices])[
        :, None
    ]
    # Config's object cap supplies a fixed padded N; the active count remains variable.
    masks = np.zeros((observed_count, p.max_objects, size, size), dtype=bool)
    visibility = np.zeros((observed_count, p.max_objects), dtype=np.float32)
    for row, frame_index in enumerate(observed_indices):
        frame_masks, frame_visibility = rendered[frame_index][1:]
        masks[row, :count] = frame_masks
        visibility[row, :count] = frame_visibility

    actions = np.zeros((observed_count, m.action_dim), dtype=np.float32)
    robots = np.zeros((observed_count, m.robot_dim), dtype=np.float32)
    observed_results = np.zeros((observed_count, m.outcome_dim), dtype=np.float32)
    result_valid = np.zeros((observed_count, m.outcome_dim), dtype=bool)
    anchor_poses = np.zeros((observed_count, 2, 4, 4), dtype=np.float32)
    timestamps = capture_times[observed_indices].astype(np.float32)
    arrival_times = arrivals[observed_indices].astype(np.float32)
    segment_count = min(observed_count, m.max_segments)
    segment_ids = np.arange(observed_count, dtype=np.int64) * segment_count // observed_count
    source_ids = np.asarray(
        [stable_source_id(system_id, episode_id, frame_index) for frame_index in observed_indices],
        dtype=np.int64,
    )
    prior_index = 0
    for row, frame_index in enumerate(observed_indices):
        anchor_poses[row, 0] = np.eye(4, dtype=np.float32)
        gripper = _rotation_matrix_yaw(float(states[frame_index, actor, 5]))
        gripper[:3, 3] = states[frame_index, actor, :3]
        gripper[2, 3] += max(float(radii[actor]), 0.02 * p.position_range)
        anchor_poses[row, 1] = gripper
        if row:
            start = prior_index
            action_span = commands[start:frame_index]
            if action_span.size:
                summed = action_span.sum(axis=0)
                actions[row] = summed
                if m.action_dim > 6:
                    actions[row, 6] = action_span[-1, 6]
                if m.action_dim > 7:
                    actions[row, 7] = action_span[-1, 7]
                delta = states[frame_index, actor] - states[prior_index, actor]
                active_outcomes = min(6, m.outcome_dim)
                observed_results[row, :active_outcomes] = delta[:active_outcomes]
                result_valid[row, :active_outcomes] = True
            prior_index = frame_index
        robot_state = _robot_features(states, frame_index, actor, commands, config, radii[actor])
        robots[row, : min(m.robot_dim, robot_state.size)] = robot_state[
            : min(m.robot_dim, robot_state.size)
        ]

    task = _make_task(m.task_dim, colors[actor], goal)
    object_slots = p.max_objects
    poses = np.zeros((object_slots, 6), dtype=np.float32)
    pose_valid = np.zeros((object_slots, 6), dtype=bool)
    entity_valid = np.zeros((object_slots,), dtype=bool)
    poses[:count] = states[last_capture, :count]
    pose_valid[:count] = True
    entity_valid[:count] = True

    change_times = np.zeros((object_slots, object_slots), dtype=np.int64)
    change_valid = np.zeros((object_slots, object_slots), dtype=bool)
    for left in range(count):
        for right in range(left + 1, count):
            change_valid[left, right] = change_valid[right, left] = True
    if follower is not None and activation_step is not None:
        if activation_step == 0:
            event_label = 1
        elif activation_step <= last_capture:
            bin_index = min(
                m.time_bins - 1, int((activation_step / max(1, last_capture)) * m.time_bins)
            )
            event_label = 2 + bin_index
        else:
            event_label = 0  # relation change is outside the observed temporal window
        change_times[actor, follower] = change_times[follower, actor] = event_label

    latest_pose = states[last_capture, :count]
    image_points = np.zeros((object_slots, 2), dtype=np.float32)
    image_points_valid = np.zeros((object_slots,), dtype=bool)
    for entity, pose in enumerate(latest_pose):
        u, v, depth = _project_world_point(pose, intrinsics, camera_to_base)
        if math.isfinite(u) and math.isfinite(v) and depth > 0:
            image_points[entity] = (u, v)
            image_points_valid[entity] = True

    current_robot = _robot_features(states, last_capture, actor, commands, config, radii[actor])
    next_robot = _robot_features(states, last_capture + 1, actor, commands, config, radii[actor])
    future_response = np.zeros((m.robot_dim,), dtype=np.float32)
    future_response_valid = np.zeros((m.robot_dim,), dtype=bool)
    response_dim = min(m.robot_dim, current_robot.size)
    future_response[:response_dim] = next_robot[:response_dim] - current_robot[:response_dim]
    future_response_valid[:response_dim] = True
    future_success = bool(
        np.linalg.norm(states[last_capture + 1, actor, :2] - goal[:2]) <= 0.15 * p.position_range
    )

    sensor = {
        "images": images.astype(np.uint8),
        "pixel_valid": np.ones((observed_count, 1, size, size), dtype=bool),
        "frame_valid": np.ones((observed_count,), dtype=bool),
        "camera_valid": np.ones((observed_count, 1), dtype=bool),
        "timestamps": timestamps,
        "arrival_times": arrival_times,
        "query_time": np.asarray(query_time, dtype=np.float32),
        "actions": actions,
        "robots": robots,
        "calibration": np.broadcast_to(calibration, (observed_count, 25)).copy()[:, None],
        "task": task,
        "observed_results": observed_results,
        "result_valid": result_valid,
        "segment_ids": segment_ids,
        "anchor_poses": anchor_poses,
        "metric_valid": np.asarray(True, dtype=bool),
        "source_ids": source_ids,
        "reference_version": np.asarray(0, dtype=np.int64),
    }
    targets = {
        "poses": poses,
        "pose_valid": pose_valid,
        "instance_masks": masks,
        "entity_valid": entity_valid,
        "visibility": visibility,
        "change_times": change_times,
        "change_valid": change_valid,
        "future_actions": commands[last_capture].copy(),
        "future_response": future_response,
        "future_response_valid": future_response_valid,
        "future_image": rendered[last_capture + 1][0].transpose(2, 0, 1)[None].astype(np.uint8),
        "future_image_valid": np.ones((1,), dtype=bool),
        "future_success": np.asarray(float(future_success), dtype=np.float32),
        "future_success_valid": np.asarray(True, dtype=bool),
        "image_points": image_points,
        "image_points_valid": image_points_valid,
    }
    validate_episode_arrays(sensor, targets, config)
    return sensor, targets


def _split_systems(system_ids: list[str], config: Config) -> dict[str, list[str]]:
    rng = np.random.default_rng(stable_seed("split", config.train.seed))
    shuffled = list(system_ids)
    rng.shuffle(shuffled)
    count = len(shuffled)
    train_count = max(1, int(count * config.prior.train_fraction))
    val_count = max(1, int(count * config.prior.val_fraction))
    train_count = min(train_count, count - 2)
    val_count = min(val_count, count - train_count - 1)
    return {
        "train": sorted(shuffled[:train_count]),
        "val": sorted(shuffled[train_count : train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count :]),
    }


def generate_dataset(config: Config, output: str | Path) -> dict[str, Any]:
    """Write a deterministic procedural RGB kinematic dataset and validate its manifest."""
    _validate_generator_config(config)
    root = Path(output)
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("refusing to overwrite an unreadable dataset manifest") from exc
        if existing.get("backend") != SYNTHETIC_BACKEND:
            raise ValueError("refusing to overwrite a dataset with a different backend")

    system_ids = [f"system-{index:06d}" for index in range(config.prior.systems)]
    system_splits = _split_systems(system_ids, config)
    split_by_system = {
        system_id: split for split, members in system_splits.items() for system_id in members
    }
    episodes: list[dict[str, Any]] = []
    for system_id in system_ids:
        mechanism_seed = stable_seed("system-mechanism", config.train.seed, system_id)
        mechanism = _mechanism(mechanism_seed)
        background = _background(
            config.model.image_size, stable_seed("system-background", config.train.seed, system_id)
        )
        for episode_index in range(config.prior.episodes_per_system):
            episode_id = f"{system_id}-episode-{episode_index:06d}"
            safe_identifier(episode_id, "episode_id")
            sensor, targets = _episode_data(
                config,
                system_id=system_id,
                episode_id=episode_id,
                split=split_by_system[system_id],
                episode_seed=stable_seed("episode", config.train.seed, system_id, episode_id),
                mechanism=mechanism,
                background=background,
            )
            arrays = build_record_arrays(
                sensor,
                targets,
                system_id=system_id,
                episode_id=episode_id,
                split=split_by_system[system_id],
                backend=SYNTHETIC_BACKEND,
            )
            relative = f"episodes/{system_id}/{episode_id}.npz"
            digest = write_npz(root / Path(*relative.split("/")), arrays)
            episodes.append(
                {
                    "path": relative,
                    "split": split_by_system[system_id],
                    "system_id": system_id,
                    "episode_id": episode_id,
                    "sha256": digest,
                }
            )

    manifest = make_manifest(
        config, backend=SYNTHETIC_BACKEND, system_splits=system_splits, episodes=episodes
    )
    write_manifest(root, manifest)
    return validate_manifest(root, config)
