from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch.distributions import Categorical, MixtureSameFamily, MultivariateNormal

from rch_pfn.config import load_config
from rch_pfn.losses import lowrank_marginal_log_prob, mixture_log_prob, state_nll
from rch_pfn.schema import JointBelief, TrainingTargets

SMOKE_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "smoke.yaml"


@pytest.fixture(scope="module")
def model_cfg():
    return load_config(SMOKE_CONFIG).model


def test_lowrank_masked_marginal_matches_dense_multivariate_normal() -> None:
    mean = torch.tensor([[0.2, -0.4, 0.7, 0.1], [-0.3, 0.5, 0.2, -0.6]])
    diagonal = torch.tensor([[0.4, 0.8, 0.6, 0.5], [0.7, 0.3, 0.9, 0.6]])
    factor = torch.tensor(
        [
            [[0.2, -0.1], [0.0, 0.3], [-0.2, 0.1], [0.4, 0.2]],
            [[-0.1, 0.2], [0.3, 0.1], [0.2, -0.2], [0.1, 0.4]],
        ]
    )
    value = torch.tensor([0.1, 9.0, -0.2, 0.8])
    valid = torch.tensor([True, False, True, True])

    actual = lowrank_marginal_log_prob(mean, diagonal, factor, value, valid)
    expected = []
    for mode in range(mean.shape[0]):
        selected_factor = factor[mode, valid]
        covariance = torch.diag(diagonal[mode, valid]) + selected_factor @ selected_factor.T
        dense = MultivariateNormal(mean[mode, valid], covariance_matrix=covariance)
        expected.append(dense.log_prob(value[valid]))
    torch.testing.assert_close(actual, torch.stack(expected), atol=1e-5, rtol=1e-5)

    empty = lowrank_marginal_log_prob(mean, diagonal, factor, value, torch.zeros_like(valid))
    torch.testing.assert_close(empty, torch.zeros(mean.shape[0]))


def test_mode_mixture_is_a_normalized_global_mixture() -> None:
    dtype = torch.float64
    logits = torch.tensor([math.log(0.3), math.log(0.7)], dtype=dtype)
    means = torch.tensor([[-2.0, -2.0], [2.0, 2.0]], dtype=dtype)
    covariance = torch.eye(2, dtype=dtype).expand(2, 2, 2)
    components = MultivariateNormal(means, covariance_matrix=covariance)
    mixture = MixtureSameFamily(Categorical(logits=logits), components)
    points = torch.tensor([[-2.0, -2.0], [2.0, 2.0], [2.0, -2.0]], dtype=dtype)

    actual = mixture_log_prob(logits, components.log_prob(points[:, None, :]))
    torch.testing.assert_close(actual, mixture.log_prob(points))
    assert torch.softmax(logits, -1).sum().item() == pytest.approx(1.0)

    axis = torch.linspace(-8.0, 8.0, 161, dtype=dtype)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    grid = torch.stack((x, y), -1)
    log_density = mixture_log_prob(logits, components.log_prob(grid[..., None, :]))
    integral = torch.trapezoid(torch.trapezoid(log_density.exp(), x=axis, dim=-1), x=axis, dim=-1)
    assert integral.item() == pytest.approx(1.0, abs=2e-4)

    # The shared mode must explain the whole vector; independently mixing each
    # coordinate would assign much more density to this cross-mode combination.
    cross_mode_log_prob = actual[-1]
    aligned_log_prob = actual[:2].max()
    assert cross_mode_log_prob < aligned_log_prob - 3.0


def _state_targets(model_cfg, *, symmetric: bool) -> TrainingTargets:
    objects, h = 1, model_cfg.image_size
    equivalent_poses = torch.zeros((1, 2, objects, 6)) if symmetric else None
    if equivalent_poses is not None:
        # The equivalence class includes the canonical pose and its x-shifted representative.
        equivalent_poses[0, 1, 0, 0] = 1.0
    return TrainingTargets(
        poses=torch.zeros((1, objects, 6)),
        pose_valid=torch.ones((1, objects, 6), dtype=torch.bool),
        instance_masks=torch.zeros((1, 1, objects, h, h)),
        entity_valid=torch.ones((1, objects), dtype=torch.bool),
        visibility=torch.ones((1, 1, objects), dtype=torch.bool),
        change_times=torch.zeros((1, objects, objects), dtype=torch.long),
        change_valid=torch.zeros((1, objects, objects), dtype=torch.bool),
        future_actions=torch.zeros((1, model_cfg.action_dim)),
        future_response=torch.zeros((1, model_cfg.robot_dim)),
        future_response_valid=torch.zeros((1, model_cfg.robot_dim), dtype=torch.bool),
        future_image=torch.zeros((1, 1, 3, h, h)),
        future_image_valid=torch.zeros((1, 1), dtype=torch.bool),
        future_success=torch.zeros((1,)),
        future_success_valid=torch.zeros((1,), dtype=torch.bool),
        equivalent_poses=equivalent_poses,
        equivalent_valid=torch.ones((1, 2), dtype=torch.bool) if symmetric else None,
    )


def _two_mode_belief(model_cfg) -> JointBelief:
    slots, modes = model_cfg.slots, 2
    nodes = slots + 2
    edge_count = nodes * (nodes - 1)
    mean = torch.zeros((1, modes, slots, 6))
    # Translation is scored in the fixed chart, where one physical meter is 10.
    mean[0, 1, 0, 0] = 1.0 / model_cfg.translation_scale
    return JointBelief(
        logits=torch.zeros((1, modes)),
        mean=mean,
        diagonal=torch.ones((1, modes, slots * 6)),
        factor=torch.zeros((1, modes, slots * 6, model_cfg.covariance_rank)),
        node_features=torch.zeros((1, modes, nodes, model_cfg.dim)),
        change_logits=torch.zeros((1, modes, edge_count, model_cfg.time_bins + 2)),
        relevance_logits=torch.zeros((1, modes, edge_count)),
        contact_logits=torch.zeros((1, modes, edge_count, 3)),
    )


def test_state_nll_scores_multimodal_symmetry_as_a_union(model_cfg) -> None:
    belief = _two_mode_belief(model_cfg)
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    symmetric = _state_targets(model_cfg, symmetric=True)
    single = replace(symmetric, equivalent_poses=None, equivalent_valid=None)

    symmetry_loss, symmetry_count = state_nll(belief, symmetric, matches, model_cfg)
    single_loss, single_count = state_nll(belief, single, matches, model_cfg)

    assert symmetry_count.item() == single_count.item() == 6
    assert symmetry_loss < single_loss
    assert (single_loss - symmetry_loss).item() == pytest.approx(math.log(2) / 6, abs=1e-5)


def test_state_nll_rejects_duplicate_symmetry_representatives(model_cfg) -> None:
    belief = _two_mode_belief(model_cfg)
    targets = _state_targets(model_cfg, symmetric=True)
    targets.equivalent_poses = torch.zeros((1, 2, 1, 6))
    targets.equivalent_valid = torch.ones((1, 2), dtype=torch.bool)
    matches = [(torch.tensor([0]), torch.tensor([0]))]

    with pytest.raises(ValueError, match="duplicate symmetry alternative"):
        state_nll(belief, targets, matches, model_cfg)


def test_symmetry_marginalizes_unlabeled_coordinates_and_empty_targets(model_cfg):
    belief = _two_mode_belief(model_cfg)
    targets = _state_targets(model_cfg, symmetric=True)
    targets.pose_valid[..., 0] = False
    matches = [(torch.tensor([0]), torch.tensor([0]))]
    marginal, _ = state_nll(belief, targets, matches, model_cfg)
    canonical, _ = state_nll(
        belief, replace(targets, equivalent_poses=None, equivalent_valid=None), matches, model_cfg
    )
    torch.testing.assert_close(marginal, canonical)
    targets.pose_valid.zero_()
    empty, count = state_nll(belief, targets, matches, model_cfg)
    assert empty.item() == 0
    assert count.item() == 0
