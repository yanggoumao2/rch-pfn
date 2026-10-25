from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from rch_pfn.config import Config, ModelConfig, PriorConfig, TrainConfig
from rch_pfn.data import (
    EpisodeDataset,
    collate_episodes,
    export_recorder_npz,
    generate_dataset,
    import_recorder_npz,
    validate_manifest,
)


def tiny_config() -> Config:
    return Config(
        model=ModelConfig(
            image_size=32,
            patch_size=8,
            vision_dim=32,
            vision_layers=1,
            vision_heads=4,
            dim=32,
            heads=4,
            slots=8,
            slot_iterations=1,
            relation_layers=1,
            core_layers=1,
            graph_layers=1,
            prediction_layers=1,
            predictive_samples=2,
            ff_multiplier=2,
            modes=2,
            covariance_rank=2,
            max_cameras=1,
            max_frames=8,
            max_segments=8,
            action_dim=8,
            robot_dim=16,
            task_dim=16,
            outcome_dim=8,
            time_bins=4,
        ),
        train=TrainConfig(
            seed=127, steps=2, batch_size=2, eval_every=1, save_every=1, device="cpu"
        ),
        prior=PriorConfig(
            systems=3,
            episodes_per_system=1,
            frames=4,
            min_objects=2,
            max_objects=3,
            dt=0.1,
            position_range=0.12,
            rotation_range=0.6,
            change_probability=0.6,
            occlusion_probability=0.5,
            noise_std=0.005,
            train_fraction=0.7,
            val_fraction=0.15,
        ),
    )


def test_generation_hashes_and_system_splits_are_deterministic(tmp_path):
    config = tiny_config()
    first = generate_dataset(config, tmp_path / "first")
    second = generate_dataset(config, tmp_path / "second")

    assert first["episodes"] == second["episodes"]
    groups = first["system_splits"]
    assert set(groups) == {"train", "val", "test"}
    assert all(groups.values())
    assert not (set(groups["train"]) & set(groups["val"]))
    assert not (set(groups["train"]) & set(groups["test"]))
    assert not (set(groups["val"]) & set(groups["test"]))
    assert validate_manifest(tmp_path / "first", config)["backend"] == "procedural_rgb_kinematic"


def test_episode_round_trip_keeps_future_targets_out_of_evidence_and_points_are_projection(
    tmp_path,
):
    config = tiny_config()
    root = tmp_path / "dataset"
    generate_dataset(config, root)
    dataset = EpisodeDataset(root, "train", config)
    evidence, targets = dataset[0]

    assert evidence.images.ndim == 6 and evidence.images.shape[0] == 1
    assert targets.future_actions.shape == (1, config.model.action_dim)
    assert not hasattr(evidence, "future_actions")
    assert targets.image_points is not None and targets.image_points_valid is not None

    calibration = evidence.calibration[0, -1, 0].numpy()
    intrinsics = calibration[:9].reshape(3, 3)
    camera_to_base = calibration[9:].reshape(4, 4)
    base_to_camera = np.linalg.inv(camera_to_base)
    for entity in np.flatnonzero(targets.image_points_valid[0].numpy()):
        point_base = np.r_[targets.poses[0, entity, :3].numpy(), 1.0]
        point_camera = base_to_camera @ point_base
        projected = intrinsics @ point_camera[:3]
        expected = projected[:2] / projected[2]
        np.testing.assert_allclose(targets.image_points[0, entity].numpy(), expected, atol=1e-4)

    rows = [dataset[index][0:2] for index in range(len(dataset))]
    # Other splits exercise batching across records with distinct frame and object counts.
    rows.extend(EpisodeDataset(root, split, config)[0] for split in ("val", "test"))
    batched_evidence, batched_targets = collate_episodes(rows)
    assert batched_evidence.images.shape[0] == 3
    assert batched_targets.poses.shape[0] == 3


def test_recorder_npz_export_import_and_manifest_hash_rejection(tmp_path):
    config = tiny_config()
    source_root = tmp_path / "source"
    generate_dataset(config, source_root)
    evidence, targets = EpisodeDataset(source_root, "train", config)[0]
    recorder_file = tmp_path / "recording.npz"
    export_recorder_npz(
        recorder_file,
        evidence,
        targets,
        config=config,
        system_id="recorder-system",
        episode_id="capture-0001",
        split="train",
    )

    imported_root = tmp_path / "imported"
    imported = import_recorder_npz(recorder_file, imported_root, config=config)
    assert imported["backend"] == "recorder_npz_import"
    assert len(EpisodeDataset(imported_root, "train", config)) == 1

    entry = imported["episodes"][0]
    record_path = imported_root / entry["path"]
    record_path.write_bytes(record_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_manifest(imported_root, config)


def test_collate_pads_target_object_axis_and_optional_points(tmp_path):
    config = tiny_config()
    root = tmp_path / "dataset"
    generate_dataset(config, root)
    sample = EpisodeDataset(root, "train", config)[0]
    evidence, targets = sample
    keep = targets.entity_valid[0].nonzero().flatten()[:1]
    smaller = replace(
        targets,
        poses=targets.poses[:, keep],
        pose_valid=targets.pose_valid[:, keep],
        instance_masks=targets.instance_masks[:, :, keep],
        entity_valid=targets.entity_valid[:, keep],
        visibility=targets.visibility[:, :, keep],
        change_times=targets.change_times[:, keep][:, :, keep],
        change_valid=targets.change_valid[:, keep][:, :, keep],
        image_points=targets.image_points[:, keep],
        image_points_valid=targets.image_points_valid[:, keep],
    )
    result_evidence, result_targets = collate_episodes([(evidence, targets), (evidence, smaller)])
    assert result_evidence.images.shape[0] == 2
    assert result_targets.poses.shape[1] == targets.poses.shape[1]
    assert result_targets.image_points is not None


def test_optional_relation_supervision_roundtrip_and_mixed_missing(tmp_path):
    config = tiny_config()
    root = tmp_path / "data"
    generate_dataset(config, root)
    evidence, targets = EpisodeDataset(root, "train", config)[0]
    n = targets.poses.shape[1]
    labeled = replace(
        targets,
        relation_relevance=torch.zeros(1, n, n),
        relation_relevance_valid=targets.change_valid.clone(),
        contact_states=torch.ones(1, n, n, dtype=torch.long),
        contact_valid=targets.change_valid.clone(),
    )
    path = tmp_path / "record.npz"
    export_recorder_npz(
        path,
        evidence,
        labeled,
        config=config,
        system_id="labels",
        episode_id="episode",
        split="train",
    )
    imported = tmp_path / "import"
    import_recorder_npz(path, imported, config=config)
    _, restored = EpisodeDataset(imported, "train", config)[0]
    assert restored.contact_states.dtype == torch.long
    missing = replace(targets, image_points=None, image_points_valid=None)
    _, batch = collate_episodes([(evidence, restored), (evidence, missing)])
    assert batch.contact_valid[0].any()
    assert not batch.contact_valid[1].any()
    assert not batch.image_points_valid[1].any()
