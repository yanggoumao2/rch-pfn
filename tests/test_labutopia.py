from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from rch_pfn.config import load_config
from rch_pfn.data import EpisodeDataset, collate_episodes, import_labutopia_episode
from rch_pfn.data.format import read_npz, split_recorder_arrays
from rch_pfn.model import RCHPFN

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(policy: str, action_dim: int = 9):
    config = load_config(REPO_ROOT / "configs" / "smoke.yaml")
    model = replace(
        config.model,
        action_dim=action_dim,
        robot_dim=18,
        max_frames=8,
        recorder_enabled=True,
        recorder_policy=policy,
    )
    return replace(config, model=model)


def _episode_files(root: Path) -> tuple[Path, bytes, bytes]:
    episode = root / "episode-001"
    episode.mkdir(parents=True)
    body_paths = [f"/World/robot/panda_link{index}" for index in range(9)] + [
        "/World/robot/panda_hand",
        "/World/robot/panda_leftfinger",
        "/World/robot/panda_rightfinger",
        "/World/InsertionStation/SeatFrame/TubeReference",
    ]
    frames = []
    for index, phase in enumerate(("post_reset", "hold", "insert", "insert")):
        positions = [[1.0 + index * 0.01, 2.0 + body * 0.001, 3.0] for body in range(13)]
        contacts = {}
        for side in ("left", "right"):
            count = int(index == 2)
            contacts[side] = {
                "force_world_n": [1.0, 0.0, 0.0] if count else [0.0, 0.0, 0.0],
                "normal_force_sum_n": float(count),
                "points_world_m": [[1.25, 2.5, 3.75]] if count else [],
                "normals_world": [[0.0, 0.0, 1.0]] if count else [],
                "separations_m": [0.001] if count else [],
                "count": count,
            }
        frames.append(
            {
                "time": index * 0.01,
                "phase": phase,
                "positions": positions,
                "orientations_wxyz": [[1.0, 0.0, 0.0, 0.0] for _ in range(13)],
                "joint_positions": [index * 0.01] * 7 + [0.02, 0.03],
                "joint_velocities": [0.1] * 9,
                "joint_targets": [0.2] * 9,
                "applied_targets": [0.15] * 9,
                "grasp_contacts": contacts,
                "future_unknown_frame_field": {"nested": [None, "保留"]},
            }
        )
    trace = [
        {
            "physics_step": index,
            "time_s": index * 0.002,
            "phase": "insert",
            "hand_position_world_m": [1.0 + index * 0.001, 2.0, 3.0],
            "hand_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "tube_position_world_m": [1.1 + index * 0.001, 2.0, 3.0],
            "tube_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        }
        for index in range(16)
    ]
    motion = {
        "body_paths": body_paths,
        "frames": frames,
        "actual_pose_samples_480hz": trace,
        "liner_contact_monitor_480hz": {"sample_count": 64, "by_phase": {"insert": {"contact_sample_count": 2}}},
        "unknown_root_field": {"unicode": "完整保留", "null": None},
    }
    outcome = {"experiment_success": True, "failure_reasons": [], "unknown_outcome_field": {"x": 1}}
    motion_bytes = (json.dumps(motion, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    outcome_bytes = (json.dumps(outcome, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    motion_path = episode / "motion.json"
    (episode / "outcome.json").write_bytes(outcome_bytes)
    motion_path.write_bytes(motion_bytes)
    return motion_path, motion_bytes, outcome_bytes


def _calibrated_rgb_file(
    path: Path,
    motion_path: Path,
    motion_bytes: bytes,
    outcome_bytes: bytes,
    *,
    offset_wrist_timestamp: float = 0.0,
) -> Path:
    motion = json.loads(motion_bytes)
    outcome = json.loads(outcome_bytes)
    outcome.update(success=True, task_success=True, strict_geometry_pass=True)
    outcome_bytes = (json.dumps(outcome, separators=(",", ":")) + "\n").encode()
    motion_path.with_name("outcome.json").write_bytes(outcome_bytes)
    times = np.asarray([frame["time"] for frame in motion["frames"]], dtype=np.float64)
    camera_names = ["third_person", "wrist"]
    intrinsics = np.asarray(
        [[[8.0, 0.0, 2.0], [0.0, 8.0, 2.0], [0.0, 0.0, 1.0]]] * 2,
        dtype=np.float64,
    )
    extrinsics = np.repeat(np.eye(4, dtype=np.float64)[None, None], 2, axis=0)
    extrinsics = np.repeat(extrinsics, len(times), axis=1)
    extrinsics[1, :, 0, 3] = 0.2
    rgb = np.zeros((len(times), 224, 224, 3), dtype=np.uint8)
    rgb[:, :, :, 0] = np.arange(224, dtype=np.uint8)[None, None, :] + 16
    rgb[:, :, :, 1] = np.arange(len(times), dtype=np.uint8)[:, None, None] * 16 + 32
    with h5py.File(path, "w") as h5_file:
        h5_file.attrs["schema"] = "labutopia-calibrated-rgb-v2"
        h5_file.attrs["camera_names"] = json.dumps(camera_names)
        h5_file.attrs["coordinate_convention"] = "camera_to_initial_robot_base, meters, USD camera axes"
        h5_file.attrs["timestamp_convention"] = "measured Isaac Sim camera rendering_time_s; source motion time is stored separately"
        h5_file.attrs["camera_timestamps_are_measured_rendering_time"] = True
        h5_file.attrs["capture_mode"] = "isaacsim_pose_replay"
        h5_file.attrs["source_motion_sha256"] = __import__("hashlib").sha256(motion_bytes).hexdigest()
        h5_file.attrs["source_outcome_sha256"] = __import__("hashlib").sha256(outcome_bytes).hexdigest()
        h5_file.attrs["success"] = True
        h5_file.attrs["task_success"] = True
        h5_file.attrs["strict_geometry_pass"] = True
        h5_file.create_dataset("timestamps", data=times)
        h5_file.create_dataset("source_timestamps", data=times)
        camera_times = np.repeat(times[None, :], 2, axis=0)
        camera_times[1, -1] += offset_wrist_timestamp
        h5_file.create_dataset("camera_timestamps", data=camera_times)
        h5_file.create_dataset(
            "agent_pose",
            data=np.asarray([frame["joint_positions"] for frame in motion["frames"]], dtype=np.float32),
        )
        h5_file.create_dataset(
            "actions",
            data=np.asarray([frame["applied_targets"] for frame in motion["frames"]], dtype=np.float32),
        )
        h5_file.create_dataset("source_frame_indices", data=np.arange(len(times), dtype=np.int64))
        h5_file.create_dataset("camera_valid", data=np.ones((2, len(times)), dtype=bool))
        h5_file.create_dataset("camera_intrinsics", data=intrinsics)
        h5_file.create_dataset("camera_extrinsics", data=extrinsics)
        h5_file.create_dataset("camera_distortion", data=np.zeros((2, 5), dtype=np.float64))
        h5_file.create_dataset("render_sequence_numbers", data=np.tile(np.arange(len(times)) + 1, (2, 1)))
        reference_numerators = np.rint(times * 100).astype(np.int64)
        h5_file.create_dataset(
            "render_frame_references",
            data=np.stack(
                [np.stack([reference_numerators, np.full_like(reference_numerators, 100)], axis=-1)] * 2
            ),
        )
        for index, name in enumerate(camera_names):
            image_data = rgb.copy()
            image_data[:, :, :, 2] = (index + 1) * 32
            h5_file.create_dataset(f"{name}_rgb", data=image_data)
    return path


def test_labutopia_import_preserves_all_source_json_and_projects_expected_channels(tmp_path):
    motion_path, motion_bytes, outcome_bytes = _episode_files(tmp_path)
    config = _config("recorder_oracle")
    output = tmp_path / "dataset"

    result = import_labutopia_episode(
        motion_path,
        output,
        config=config,
        system_id="system-001",
        episode_id="episode-001",
        policy="recorder_oracle",
    )

    record = output / result["episode"]
    recorder, archives = split_recorder_arrays(read_npz(record))
    assert np.array_equal(archives["motion_json"], np.frombuffer(motion_bytes, dtype=np.uint8))
    assert np.array_equal(archives["outcome_json"], np.frombuffer(outcome_bytes, dtype=np.uint8))
    assert recorder["body_positions"].shape == (4, 13, 3)
    assert recorder["joint_positions"].shape == (4, 9)
    assert recorder["trace_times"].shape == (16,)
    assert recorder["body_valid"].all()
    assert recorder["contact_valid"][2].all()
    np.testing.assert_allclose(recorder["contact_points_base"][2, 0, 0], [0.25, 0.5, 0.75])
    field_manifest = json.loads(str(archives["field_manifest_json"].item()))
    assert field_manifest["all_fields_preserved"]
    assert field_manifest["field_disposition"]["raw_source_preserved"]
    assert field_manifest["normalized_coordinate_frame"] == "initial_robot_base"
    assert any(
        item["path"] == "/unknown_root_field/unicode" and item["disposition"] == "archive_only"
        for item in field_manifest["all_source_fields"]
    )
    assert any(
        item["path"] == "/frames/*/joint_positions"
        and item["disposition"] == "model_input"
        for item in field_manifest["all_source_fields"]
    )

    evidence, targets = EpisodeDataset(output, "train", config)[0]
    assert evidence.recorder is not None
    assert evidence.recorder.body_valid.all()
    assert evidence.robots.shape[-1] == 18
    assert evidence.actions.shape[-1] == 9
    assert targets.future_success_valid.item()
    batch, _ = collate_episodes([(evidence, targets), (evidence, targets)])
    with torch.no_grad():
        prediction = RCHPFN(config.model).eval()(batch)
    assert torch.isfinite(prediction.final.mean).all()
    misaligned = replace(
        evidence.recorder,
        sample_time=evidence.recorder.sample_time + 0.001,
    )
    with pytest.raises(ValueError, match="aligned to evidence timestamps"):
        replace(evidence, recorder=misaligned).validate(config.model)


def test_deployable_policy_masks_simulator_truth_but_keeps_robot_channels(tmp_path):
    motion_path, _, _ = _episode_files(tmp_path)
    config = _config("deployable")
    output = tmp_path / "dataset"
    result = import_labutopia_episode(
        motion_path,
        output,
        config=config,
        system_id="system-001",
        episode_id="episode-001",
        policy="deployable",
    )
    recorder, _ = split_recorder_arrays(read_npz(output / result["episode"]))
    assert not recorder["body_valid"].any()
    assert not recorder["contact_valid"].any()
    assert not recorder["phase_valid"].any()
    assert not recorder["trace_valid"].any()
    assert recorder["joint_valid"].all()

    evidence, _ = EpisodeDataset(output, "train", config)[0]
    assert evidence.recorder is not None
    assert torch.equal(evidence.anchor_poses, torch.eye(4).reshape(1, 1, 1, 4, 4).expand(1, 4, 2, 4, 4))
    torch.testing.assert_close(evidence.robots[0, :, :9], evidence.recorder.joint_positions[0])
    torch.testing.assert_close(evidence.robots[0, :, 9:], evidence.recorder.joint_velocities[0])
    with torch.no_grad():
        prediction = RCHPFN(config.model).eval()(evidence)
    assert torch.isfinite(prediction.final.mean).all()
    model = RCHPFN(config.model).eval()
    for field_name in (
        "body_valid",
        "contact_valid",
        "contact_point_valid",
        "phase_valid",
        "trace_valid",
    ):
        mask = getattr(evidence.recorder, field_name).clone()
        mask.reshape(-1)[0] = True
        privileged = replace(evidence.recorder, **{field_name: mask})
        forged = replace(evidence, recorder=privileged)
        with pytest.raises(ValueError, match="deployable recorder policy forbids"):
            forged.validate(config.model)
        with pytest.raises(ValueError, match="deployable recorder policy forbids"):
            model(forged)
    with pytest.raises(ValueError, match="manifest recorder policy"):
        EpisodeDataset(output, "train", _config("recorder_oracle"))


def test_labutopia_import_supports_action_dimensions_larger_than_joint_count(tmp_path):
    motion_path, _, _ = _episode_files(tmp_path)
    config = _config("deployable", action_dim=16)
    output = tmp_path / "dataset"
    import_labutopia_episode(
        motion_path,
        output,
        config=config,
        system_id="system-001",
        episode_id="episode-001",
        policy="deployable",
    )

    evidence, _ = EpisodeDataset(output, "train", config)[0]
    assert evidence.actions.shape == (1, 4, 16)
    torch.testing.assert_close(evidence.actions[0, :, :9], torch.full((4, 9), 0.15))
    assert torch.count_nonzero(evidence.actions[0, :, 9:]) == 0


def test_calibrated_rgb_import_preserves_two_views_and_aligned_calibration(tmp_path):
    motion_path, motion_bytes, outcome_bytes = _episode_files(tmp_path)
    rgb_path = _calibrated_rgb_file(
        tmp_path / "episode.h5", motion_path, motion_bytes, outcome_bytes
    )
    config = _config("deployable")
    output = tmp_path / "dataset-rgb"
    result = import_labutopia_episode(
        motion_path,
        output,
        config=config,
        system_id="system-rgb",
        episode_id="episode-rgb",
        calibrated_rgb_path=rgb_path,
    )

    arrays = read_npz(output / result["episode"])
    assert arrays["sensor__images"].shape == (4, 2, 3, config.model.image_size, config.model.image_size)
    assert arrays["sensor__camera_valid"].all()
    assert arrays["sensor__pixel_valid"].all()
    np.testing.assert_allclose(
        arrays["sensor__calibration"][0, 0, :9],
        [8, 0, 2, 0, 8, 2, 0, 0, 1],
    )
    np.testing.assert_allclose(arrays["sensor__calibration"][0, 1, 9 + 3], 0.2)
    np.testing.assert_allclose(arrays["sensor__timestamps"], arrays["recorder__sample_time"])
    archive_manifest = json.loads(str(arrays["archive__rgb_record_manifest_json"].item()))
    assert archive_manifest["camera_names"] == ["third_person", "wrist"]
    assert archive_manifest["selected_source_frame_indices"] == [0, 1, 2, 3]
    assert result["camera_names"] == ["third_person", "wrist"]

    evidence, _ = EpisodeDataset(output, "train", config)[0]
    assert evidence.camera_valid.all()
    assert evidence.calibration.shape == (1, 4, 2, 25)
    assert not evidence.recorder.body_valid.any()


def test_calibrated_rgb_import_rejects_timestamp_misalignment(tmp_path):
    motion_path, motion_bytes, outcome_bytes = _episode_files(tmp_path)
    rgb_path = _calibrated_rgb_file(
        tmp_path / "misaligned.h5",
        motion_path,
        motion_bytes,
        outcome_bytes,
        offset_wrist_timestamp=1e-4,
    )
    with pytest.raises(ValueError, match="camera timestamps are not aligned"):
        import_labutopia_episode(
            motion_path,
            tmp_path / "dataset-rgb",
            config=_config("deployable"),
            calibrated_rgb_path=rgb_path,
        )


def test_import_rejects_unregistered_phase_instead_of_hash_aliasing(tmp_path):
    motion_path, _, _ = _episode_files(tmp_path)
    motion = json.loads(motion_path.read_text(encoding="utf-8"))
    motion["frames"][2]["phase"] = "custom_future_phase"
    motion_path.write_text(json.dumps(motion), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported LabUtopia phase"):
        import_labutopia_episode(
            motion_path,
            tmp_path / "dataset",
            config=_config("recorder_oracle"),
            policy="recorder_oracle",
        )
