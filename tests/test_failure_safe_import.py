import pytest

from rch_pfn.data.labutopia import validate_calibrated_rgb_outcome_metadata

VALID_SUCCESS = {
    "success": True,
    "task_success": True,
    "strict_geometry_pass": True,
}


def test_default_import_rejects_non_success_outcome():
    outcome = {**VALID_SUCCESS, "success": False}

    with pytest.raises(ValueError, match="requires source outcome success=true"):
        validate_calibrated_rgb_outcome_metadata(outcome, outcome, "success")


def test_failure_safe_import_accepts_matching_non_success_outcome():
    outcome = {**VALID_SUCCESS, "success": False, "task_success": False}

    checked = validate_calibrated_rgb_outcome_metadata(outcome, outcome, "failure-safe")

    assert checked is outcome


def test_failure_safe_import_rejects_recorder_outcome_mismatch():
    outcome = {**VALID_SUCCESS, "success": False, "task_success": False}
    recorded = {**outcome, "task_success": True}

    with pytest.raises(ValueError, match="does not match source outcome"):
        validate_calibrated_rgb_outcome_metadata(outcome, recorded, "failure-safe")
