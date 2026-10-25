"""One full-architecture forward pass on workstation CPU; not a training benchmark."""

import json
import time

import torch

from rch_pfn.config import load_config
from rch_pfn.model import RCHPFN
from rch_pfn.schema import EvidenceBatch


def main():
    torch.set_num_threads(4)
    cfg = load_config("configs/default.yaml").model
    h = cfg.image_size
    torch.manual_seed(7)
    model = RCHPFN(cfg).eval()
    e = EvidenceBatch(
        images=torch.rand(1, 2, 1, 3, h, h),
        pixel_valid=torch.ones(1, 2, 1, h, h, dtype=torch.bool),
        frame_valid=torch.ones(1, 2, dtype=torch.bool),
        camera_valid=torch.ones(1, 2, 1, dtype=torch.bool),
        timestamps=torch.tensor([[0.0, 1.0]]),
        arrival_times=torch.tensor([[0.0, 1.0]]),
        query_time=torch.tensor([1.0]),
        actions=torch.zeros(1, 2, cfg.action_dim),
        robots=torch.zeros(1, 2, cfg.robot_dim),
        calibration=torch.zeros(1, 2, 1, 25),
        task=torch.zeros(1, cfg.task_dim),
        observed_results=torch.zeros(1, 2, cfg.outcome_dim),
        result_valid=torch.zeros(1, 2, cfg.outcome_dim, dtype=torch.bool),
        segment_ids=torch.tensor([[0, 1]]),
        anchor_poses=torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1),
        metric_valid=torch.zeros(1, dtype=torch.bool),
        source_ids=torch.tensor([[1, 2]]),
        reference_version=torch.zeros(1, dtype=torch.long),
    )
    started = time.perf_counter()
    with torch.no_grad():
        result = model(e, torch.zeros(1, cfg.action_dim))
    assert result.final.mean.shape == (1, 8, 32, 6)
    assert torch.isfinite(result.final.mean).all()
    assert torch.isfinite(result.final.factor).all()
    assert torch.isfinite(result.prediction["response_mean"]).all()
    print(
        json.dumps(
            {
                "parameters": sum(p.numel() for p in model.parameters()),
                "mean_shape": list(result.final.mean.shape),
                "directed_edges": result.edge_index.shape[1],
                "elapsed_seconds": time.perf_counter() - started,
                "device": "cpu",
                "finite": True,
            }
        )
    )


if __name__ == "__main__":
    main()
