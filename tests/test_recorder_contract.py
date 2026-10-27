import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "record_labutopia_pose_replay.py"
_SPEC = importlib.util.spec_from_file_location("record_labutopia_pose_replay", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


VALID_SUCCESS = {
    "success": True,
    "task_success": True,
    "strict_geometry_pass": True,
}


def test_success_capture_rejects_non_success_outcome():
    outcome = {**VALID_SUCCESS, "task_success": False, "strict_geometry_pass": False}

    with pytest.raises(ValueError, match="verified strict-geometry success"):
        _MODULE.validate_source_outcome(outcome, "success")


def test_failure_safe_capture_accepts_and_preserves_non_success_outcome():
    outcome = {**VALID_SUCCESS, "success": False, "task_success": False}

    checked = _MODULE.validate_source_outcome(outcome, "failure-safe")

    assert checked is outcome
    assert checked["success"] is False
    assert checked["task_success"] is False
    assert checked["strict_geometry_pass"] is True


def test_failure_safe_capture_rejects_missing_or_non_boolean_status():
    with pytest.raises(ValueError, match="boolean"):
        _MODULE.validate_source_outcome(
            {"success": False, "task_success": False}, "failure-safe"
        )

    with pytest.raises(ValueError, match="boolean"):
        _MODULE.validate_source_outcome(
            {**VALID_SUCCESS, "success": None}, "failure-safe"
        )
