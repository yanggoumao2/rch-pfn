from dataclasses import fields, replace

import pytest
import torch

from rch_pfn.config import load_config
from rch_pfn.inference import StreamingSession
from rch_pfn.model import RCHPFN
from rch_pfn.schema import EvidenceBatch


def frame(cfg, source, segment):
    h = cfg.image_size
    pose = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, 2, 1, 1)
    return EvidenceBatch(
        images=torch.zeros(1, 1, 1, 3, h, h),
        pixel_valid=torch.ones(1, 1, 1, h, h, dtype=torch.bool),
        frame_valid=torch.ones(1, 1, dtype=torch.bool),
        camera_valid=torch.ones(1, 1, 1, dtype=torch.bool),
        timestamps=torch.tensor([[float(source)]]),
        arrival_times=torch.tensor([[float(source)]]),
        query_time=torch.tensor([float(source)]),
        actions=torch.zeros(1, 1, cfg.action_dim),
        robots=torch.zeros(1, 1, cfg.robot_dim),
        calibration=torch.zeros(1, 1, 1, 25),
        task=torch.zeros(1, cfg.task_dim),
        observed_results=torch.zeros(1, 1, cfg.outcome_dim),
        result_valid=torch.zeros(1, 1, cfg.outcome_dim, dtype=torch.bool),
        segment_ids=torch.tensor([[segment]]),
        anchor_poses=pose,
        metric_valid=torch.zeros(1, dtype=torch.bool),
        source_ids=torch.tensor([[source]]),
        reference_version=torch.zeros(1, dtype=torch.long),
    )


def test_stable_segment_identity_survives_eviction():
    cfg = load_config("configs/smoke.yaml").model
    cfg.max_segments = 2
    session = StreamingSession(RCHPFN(cfg))
    for source, segment in [(1, 100), (2, 101), (3, 102), (4, 102)]:
        session.append(frame(cfg, source, segment))
    assert session.evidence.segment_ids.tolist() == [[0, 1, 1]]
    assert session.evidence.source_ids.tolist() == [[2, 3, 4]]
    assert session._external_segments.tolist() == [[101, 102, 102]]


def test_padding_is_not_a_segment_and_rejected_append_is_transactional():
    cfg = load_config("configs/smoke.yaml").model
    cfg.max_segments = 2
    session = StreamingSession(RCHPFN(cfg))
    item = frame(cfg, 1, 100)
    temporal = {f.name for f in fields(EvidenceBatch)} - {
        "query_time",
        "task",
        "metric_valid",
        "reference_version",
    }
    values = {
        name: torch.cat([getattr(item, name), torch.zeros_like(getattr(item, name))], 1)
        for name in temporal
    }
    session.append(replace(item, **values))
    assert session.evidence.segment_ids.tolist() == [[0]]
    before = session.evidence
    external = session._external_segments.clone()
    bad = replace(frame(cfg, 2, 101), reference_version=torch.ones(1, dtype=torch.long))
    with pytest.raises(ValueError, match="reference changed"):
        session.append(bad)
    assert session.evidence is before
    torch.testing.assert_close(session._external_segments, external)
    session.append(frame(cfg, 2, 101))
    session.append(frame(cfg, 3, 102))
    assert session.evidence.source_ids.tolist() == [[2, 3]]
