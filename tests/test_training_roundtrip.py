import copy

import torch

from rch_pfn.config import load_config
from rch_pfn.data import EpisodeDataset, generate_dataset
from rch_pfn.model import RCHPFN
from rch_pfn.runner import evaluate, evaluate_model, load_checkpoint, train


def test_train_resume_is_identical_and_test_split_evaluates(tmp_path):
    cfg = load_config("configs/smoke.yaml")
    cfg.prior.systems = 6
    cfg.prior.episodes_per_system = 1
    cfg.train.steps = 2
    cfg.train.eval_every = 2
    cfg.train.save_every = 1
    cfg.train.device = "cpu"
    data = tmp_path / "data"
    generate_dataset(cfg, data)
    full = tmp_path / "full"
    interrupted = tmp_path / "interrupted"
    train(cfg, data, full)
    one = copy.deepcopy(cfg)
    one.train.steps = 1
    train(one, data, interrupted)
    # A lock file left by a dead process does not own the OS lock.
    (interrupted / ".training.lock").write_text("orphaned marker")
    train(cfg, data, interrupted, resume=interrupted / "last.pt")
    continuous = load_checkpoint(full / "last.pt")
    resumed = load_checkpoint(interrupted / "last.pt")
    assert continuous["step"] == resumed["step"] == 2
    for name, parameter in continuous["model"].items():
        torch.testing.assert_close(parameter, resumed["model"][name], rtol=0, atol=0)
    report = evaluate(interrupted / "last.pt", data, "test", "cpu")
    assert report["examples"] > 0
    assert report["physical_closed_loop_evaluation"] is False
    assert report["calibrated"] is False
    model = RCHPFN(cfg.model)
    model.load_state_dict(resumed["model"])
    teacher = copy.deepcopy(model.visual).eval().requires_grad_(False)
    teacher.load_state_dict(resumed["teacher"])
    dataset = EpisodeDataset(data, "train", cfg)
    single = copy.deepcopy(cfg)
    single.train.batch_size = 1
    grouped = copy.deepcopy(cfg)
    grouped.train.batch_size = 3
    first = evaluate_model(model, teacher, dataset, single, torch.device("cpu"))
    second = evaluate_model(model, teacher, dataset, grouped, torch.device("cpu"))
    for key in first:
        if key != "elapsed_seconds":
            assert first[key] == second[key], key
    expected_dimensions = sum(int(dataset[i][1].pose_valid.sum()) for i in range(len(dataset)))
    assert first["state_dimensions"] == expected_dimensions
