from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from rch_pfn.control import (
    BackendStepResult,
    BoundedStep,
    ControlSession,
    ControlState,
    DryRunBackend,
    HeldObject,
    PlanReply,
    PosteriorMode,
    RepairSpec,
    command_fingerprint,
    gate_plan,
    prepare_bounded_step,
)
from rch_pfn.geometry import (
    compose_gripper_target,
    invert_transform,
    matrix_to_pose,
    matrix_to_rotvec,
    pose_to_matrix,
    relative_transform,
    rotvec_to_matrix,
)


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> tuple[float, ...]:
    return (x, y, z, 0.0, 0.0, 0.0)


def _spec(**changes: object) -> RepairSpec:
    base = RepairSpec(
        request_id="repair-7",
        object_id="object-2",
        target_id="target-4",
        reference_version=12,
        frame_id="base",
        created_at_s=0.0,
        expires_at_s=10.0,
        target_pose=_pose(),
        desired_object_in_target=_pose(),
        object_in_gripper=_pose(0.1),
    )
    return replace(base, **changes)


def _command(
    spec: RepairSpec, *, start: tuple[float, ...] | None = None, action_id: str = "step-7"
) -> BoundedStep:
    return prepare_bounded_step(
        spec,
        action_id=action_id,
        current_gripper_pose=_pose(-0.105) if start is None else start,
        robot_state_version=4,
        held_object=HeldObject(spec.object_id, "test-object-shape", 1, spec.object_in_gripper),
    )


def _reply(spec: RepairSpec, command: BoundedStep | None = None, **changes: object) -> PlanReply:
    command = _command(spec) if command is None else command
    base = PlanReply(
        request_id=spec.request_id,
        object_id=spec.object_id,
        target_id=spec.target_id,
        reference_version=spec.reference_version,
        frame_id=spec.frame_id,
        generated_at_s=0.9,
        identity_confidence=0.999,
        calibrated=True,
        metric_valid=True,
        posterior_modes=(PosteriorMode("main", 1.0, torch.tensor([_pose()])),),
        action_id=command.action_id,
        command_fingerprint=command_fingerprint(command),
        start_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
    )
    return replace(base, **changes)


def _gate(
    spec: RepairSpec, reply: PlanReply, now_s: float = 1.0, command: BoundedStep | None = None
):
    command = _command(spec) if command is None else command
    return gate_plan(
        spec,
        reply,
        now_s,
        command=command,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
    )


def _accept(session: ControlSession, now_s: float = 1.0):
    return session.accept(
        now_s,
        current_gripper_pose=session.command.start_pose,
        robot_state_version=session.command.robot_state_version,
    )


def test_rotvec_round_trip_at_zero_small_angles_and_near_pi() -> None:
    axis = torch.tensor([0.3, -0.4, 0.5], dtype=torch.float64)
    axis = axis / torch.linalg.vector_norm(axis)
    vectors = torch.stack(
        (
            torch.zeros(3, dtype=torch.float64),
            torch.tensor([1e-10, -2e-10, 3e-10], dtype=torch.float64),
            torch.tensor([1e-5, 2e-5, -3e-5], dtype=torch.float64),
            axis * (math.pi - 1e-8),
            torch.tensor([math.pi, 0.0, 0.0], dtype=torch.float64),
        )
    )
    rotations = rotvec_to_matrix(vectors)
    recovered = matrix_to_rotvec(rotations)
    torch.testing.assert_close(rotvec_to_matrix(recovered), rotations, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(recovered).all()


def test_transform_inverse_relative_and_gripper_composition() -> None:
    target_pose = torch.tensor([0.5, -0.2, 0.1, 0.0, 0.0, 0.4], dtype=torch.float64)
    desired_in_target = torch.tensor([0.03, 0.01, 0.0, 0.0, 0.2, 0.0], dtype=torch.float64)
    object_in_gripper = torch.tensor([0.0, 0.0, 0.08, 0.0, 0.0, -0.1], dtype=torch.float64)
    target = pose_to_matrix(target_pose)
    desired = pose_to_matrix(desired_in_target)
    grasp = pose_to_matrix(object_in_gripper)

    torch.testing.assert_close(target @ invert_transform(target), torch.eye(4, dtype=torch.float64))
    relative = relative_transform(target, target @ desired)
    torch.testing.assert_close(relative, desired, atol=1e-10, rtol=1e-10)

    gripper_goal = compose_gripper_target(target, desired, grasp)
    torch.testing.assert_close(gripper_goal @ grasp, target @ desired, atol=1e-10, rtol=1e-10)


def test_gate_accepts_only_calibrated_metric_fresh_identity_matched_reply() -> None:
    spec = _spec()
    decision = _gate(spec, _reply(spec))
    assert decision.accepted
    assert decision.reason == "accepted"
    assert decision.posterior_support == pytest.approx(1.0)
    assert decision.gripper_target_pose is not None
    assert decision.gripper_target_pose[0] == pytest.approx(-0.1)

    assert _gate(spec, _reply(spec, calibrated=False)).reason == "uncalibrated"
    assert _gate(spec, _reply(spec, metric_valid=False)).reason == "metric_scale_invalid"
    assert (
        _gate(spec, _reply(spec, identity_confidence=0.5)).reason
        == "identity_confidence_below_threshold"
    )
    assert _gate(spec, _reply(spec, reference_version=13)).reason == "reference_version_mismatch"
    assert _gate(spec, _reply(spec), 1.6).reason == "stale_reply"


def test_gate_rejects_bimodal_mean_that_lies_on_target() -> None:
    spec = _spec()
    modes = (
        PosteriorMode("left", 0.5, torch.tensor([_pose(x=-0.2)])),
        PosteriorMode("right", 0.5, torch.tensor([_pose(x=0.2)])),
    )
    reply = _reply(spec, posterior_modes=modes)
    # The arithmetic mean is exactly the target, but neither mode has support
    # inside the 5 mm target tolerance.
    assert (torch.tensor([-0.2, 0.2]).mean()).item() == pytest.approx(0.0)
    decision = _gate(spec, reply)
    assert not decision.accepted
    assert decision.reason == "insufficient_posterior_support"
    assert decision.posterior_support == pytest.approx(0.0)


def test_gate_uses_posterior_mass_without_cross_mode_pose_averaging() -> None:
    spec = _spec()
    modes = (
        PosteriorMode("supported", 0.97, torch.tensor([_pose(), _pose(x=0.001)])),
        PosteriorMode("outlier", 0.03, torch.tensor([_pose(x=0.2)])),
    )
    decision = _gate(spec, _reply(spec, posterior_modes=modes))
    assert decision.accepted
    assert decision.selected_mode_id == "supported"
    assert decision.posterior_support == pytest.approx(0.97)


class _AlwaysReachable:
    def is_reachable(self, pose: tuple[float, ...], *, frame_id: str) -> bool:
        return frame_id == "base"


class _AlwaysCollisionFree:
    def is_path_collision_free(self, command: BoundedStep) -> bool:
        return command.frame_id == "base"


class _NeverReachable:
    def is_reachable(self, pose: tuple[float, ...], *, frame_id: str) -> bool:
        return False


class _Colliding:
    def is_path_collision_free(self, command: BoundedStep) -> bool:
        return False


def test_state_machine_revalidates_and_records_one_bounded_dry_run_step() -> None:
    spec = _spec(max_translation_step_m=0.01)
    command = _command(spec)
    session = ControlSession(spec, _reply(spec, command), command)
    assert session.state is ControlState.PROPOSED
    assert _accept(session).accepted
    assert session.state is ControlState.ACCEPTED

    backend = DryRunBackend()
    result = session.execute_one_step(
        now_s=1.1,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
        backend=backend,
        ik=_AlwaysReachable(),
        collision=_AlwaysCollisionFree(),
    )
    assert result.state is ControlState.COMPLETED
    assert result.reason == "dry_run_recorded"
    assert result.backend_result == BackendStepResult(
        True,
        "one bounded step recorded by DryRunBackend; no hardware was contacted",
        True,
    )
    assert len(backend.commands) == 1
    assert backend.commands[0].translation_step_m <= 0.01 + 1e-12
    assert backend.commands[0].rotation_step_rad <= 0.1 + 1e-12
    assert backend.commands[0] is command


def test_execution_rejects_expired_revalidation_and_ik_failure() -> None:
    spec = _spec()
    command = _command(spec)
    expired = ControlSession(spec, _reply(spec, command), command)
    _accept(expired)
    backend = DryRunBackend()
    result = expired.execute_one_step(
        now_s=10.0,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
        backend=backend,
        ik=_AlwaysReachable(),
        collision=_AlwaysCollisionFree(),
    )
    assert result.state is ControlState.EXPIRED
    assert result.reason == "expired"
    assert backend.commands == []

    unreachable = ControlSession(spec, _reply(spec, command), command)
    _accept(unreachable)
    result = unreachable.execute_one_step(
        now_s=1.1,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
        backend=backend,
        ik=_NeverReachable(),
        collision=_AlwaysCollisionFree(),
    )
    assert result.state is ControlState.REJECTED
    assert result.reason == "ik_unreachable"
    assert backend.commands == []

    colliding = ControlSession(spec, _reply(spec, command), command)
    _accept(colliding)
    result = colliding.execute_one_step(
        now_s=1.1,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
        backend=backend,
        ik=_AlwaysReachable(),
        collision=_Colliding(),
    )
    assert result.state is ControlState.REJECTED
    assert result.reason == "collision_check_failed"
    assert backend.commands == []


def test_session_rejects_illegal_state_transition() -> None:
    spec = _spec()
    command = _command(spec)
    session = ControlSession(spec, _reply(spec, command), command)
    with pytest.raises(RuntimeError, match="only an accepted"):
        session.execute_one_step(
            now_s=1.1,
            current_gripper_pose=_pose(),
            robot_state_version=command.robot_state_version,
            backend=DryRunBackend(),
            ik=_AlwaysReachable(),
            collision=_AlwaysCollisionFree(),
        )
    _accept(session)
    with pytest.raises(RuntimeError, match="only a proposed"):
        _accept(session)


def test_bounded_command_is_constructed_before_prediction() -> None:
    spec = _spec()
    command = _command(spec, start=(-0.5, 0.0, 0.0, 0.0, 0.0, 1.0))
    delta = matrix_to_pose(
        relative_transform(
            pose_to_matrix(torch.tensor(command.start_pose, dtype=torch.float64)),
            pose_to_matrix(torch.tensor(command.target_pose, dtype=torch.float64)),
        )
    )
    assert float(torch.linalg.vector_norm(delta[:3])) == pytest.approx(0.01)
    assert float(torch.linalg.vector_norm(delta[3:])) == pytest.approx(0.1)
    assert command.held_object.object_id == spec.object_id
    assert command.held_object.object_in_gripper == spec.object_in_gripper


def test_full_repair_posterior_cannot_authorize_truncated_command() -> None:
    spec = _spec()
    full_command = _command(replace(spec, max_translation_step_m=1.0), start=_pose(-0.5))
    bounded_command = _command(spec, start=_pose(-0.5))
    assert full_command.target_pose[0] == pytest.approx(-0.1)
    assert bounded_command.target_pose[0] == pytest.approx(-0.49)

    full_reply = _reply(spec, full_command)
    decision = _gate(spec, full_reply, command=bounded_command)
    assert not decision.accepted
    assert decision.reason == "command_fingerprint_mismatch"
    session = ControlSession(spec, full_reply, bounded_command)
    assert not _accept(session).accepted
    assert session.state is ControlState.REJECTED

    # A posterior for the actual short action predicts an object still far
    # from the repair goal and therefore fails the existing tolerance gate.
    result_pose = matrix_to_pose(
        pose_to_matrix(torch.tensor(bounded_command.target_pose, dtype=torch.float64))
        @ pose_to_matrix(torch.tensor(spec.object_in_gripper, dtype=torch.float64))
    )
    bounded_reply = _reply(
        spec,
        bounded_command,
        posterior_modes=(PosteriorMode("short-step-result", 1.0, result_pose[None]),),
    )
    decision = _gate(spec, bounded_reply, command=bounded_command)
    assert decision.reason == "insufficient_posterior_support"
    assert decision.posterior_support == 0.0


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("action_id", "another-step", "action_id_mismatch"),
        ("command_fingerprint", "not-the-command", "command_fingerprint_mismatch"),
        ("start_pose", _pose(-0.106), "reply_start_pose_mismatch"),
        ("robot_state_version", 5, "robot_state_version_mismatch"),
    ],
)
def test_gate_rejects_reply_with_wrong_action_binding(field, value, reason) -> None:
    spec = _spec()
    command = _command(spec)
    reply = _reply(spec, command, **{field: value})
    decision = _gate(spec, reply, command=command)
    assert not decision.accepted
    assert decision.reason == reason


@pytest.mark.parametrize(
    ("pose", "version", "reason"),
    [
        (_pose(-0.106), 4, "current_start_pose_mismatch"),
        (_pose(-0.105), 5, "robot_state_version_mismatch"),
    ],
)
def test_gate_and_execution_reject_changed_live_start_or_robot_state(pose, version, reason) -> None:
    spec = _spec()
    command = _command(spec)
    reply = _reply(spec, command)
    decision = gate_plan(
        spec,
        reply,
        1.0,
        command=command,
        current_gripper_pose=pose,
        robot_state_version=version,
    )
    assert decision.reason == reason

    session = ControlSession(spec, reply, command)
    assert _accept(session).accepted
    backend = DryRunBackend()
    result = session.execute_one_step(
        now_s=1.1,
        current_gripper_pose=pose,
        robot_state_version=version,
        backend=backend,
        ik=_AlwaysReachable(),
        collision=_AlwaysCollisionFree(),
    )
    assert result.state is ControlState.REJECTED
    assert result.reason == reason
    assert backend.commands == []


def test_accepted_command_cannot_be_replaced_even_with_a_rebound_reply() -> None:
    spec = _spec()
    command = _command(spec)
    session = ControlSession(spec, _reply(spec, command), command)
    assert _accept(session).accepted
    changed = replace(command, target_pose=_pose(-0.101), translation_step_m=0.004)
    session.command = changed
    session.reply = _reply(spec, changed)
    backend = DryRunBackend()
    result = session.execute_one_step(
        now_s=1.1,
        current_gripper_pose=changed.start_pose,
        robot_state_version=changed.robot_state_version,
        backend=backend,
        ik=_AlwaysReachable(),
        collision=_AlwaysCollisionFree(),
    )
    assert result.state is ControlState.REJECTED
    assert result.reason == "accepted_command_changed"
    assert backend.commands == []


def test_held_geometry_version_is_part_of_command_binding() -> None:
    spec = _spec()
    command = _command(spec)
    changed = replace(command, held_object=replace(command.held_object, geometry_version=2))
    assert command_fingerprint(command) != command_fingerprint(changed)
    assert (
        _gate(spec, _reply(spec, command), command=changed).reason == "command_fingerprint_mismatch"
    )


class _ThinSlabFixture:
    """Exact swept interval check for this fixed-orientation, straight-x fixture."""

    def __init__(self) -> None:
        self.commands: list[BoundedStep] = []
        self.endpoints_safe = False

    def is_path_collision_free(self, command: BoundedStep) -> bool:
        self.commands.append(command)
        held = command.held_object
        if (held.geometry_id, held.geometry_version) != ("test-object-shape", 1):
            return False
        if any(command.start_pose[3:]) or any(command.target_pose[3:]):
            return False
        # The attached object is a small sphere; the gripper itself is far
        # from this thin slab. Sweep the sphere, not just the gripper endpoint.
        start_x = command.start_pose[0] + held.object_in_gripper[0]
        end_x = command.target_pose[0] + held.object_in_gripper[0]
        radius, slab_min, slab_max = 0.0001, -0.0026, -0.0024
        self.endpoints_safe = all(
            x + radius < slab_min or x - radius > slab_max for x in (start_x, end_x)
        )
        sweep_min, sweep_max = min(start_x, end_x) - radius, max(start_x, end_x) + radius
        return sweep_max < slab_min or sweep_min > slab_max


def test_safe_endpoints_do_not_allow_held_object_to_cross_thin_obstacle() -> None:
    spec = _spec()
    command = _command(spec)
    session = ControlSession(spec, _reply(spec, command), command)
    assert _accept(session).accepted
    checker, backend = _ThinSlabFixture(), DryRunBackend()
    result = session.execute_one_step(
        now_s=1.1,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
        backend=backend,
        ik=_AlwaysReachable(),
        collision=checker,
    )
    assert checker.endpoints_safe
    assert checker.commands == [command]
    assert checker.commands[0] is command
    assert result.state is ControlState.REJECTED
    assert result.reason == "collision_check_failed"
    assert backend.commands == []


def test_endpoint_only_collision_checker_fails_closed() -> None:
    class EndpointOnly:
        def is_collision_free(self, pose, *, frame_id):
            raise AssertionError("the old endpoint-only method must not be used")

    spec = _spec()
    command = _command(spec)
    session = ControlSession(spec, _reply(spec, command), command)
    assert _accept(session).accepted
    backend = DryRunBackend()
    result = session.execute_one_step(
        now_s=1.1,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
        backend=backend,
        ik=_AlwaysReachable(),
        collision=EndpointOnly(),
    )
    assert result.reason == "collision_check_error"
    assert backend.commands == []


def test_non_dry_run_backend_is_never_called() -> None:
    class NotDryRun:
        is_dry_run = False

        def execute_step(self, command):
            raise AssertionError("a non-dry-run backend must not be called")

    spec = _spec()
    command = _command(spec)
    session = ControlSession(spec, _reply(spec, command), command)
    assert _accept(session).accepted
    result = session.execute_one_step(
        now_s=1.1,
        current_gripper_pose=command.start_pose,
        robot_state_version=command.robot_state_version,
        backend=NotDryRun(),
        ik=_AlwaysReachable(),
        collision=_AlwaysCollisionFree(),
    )
    assert result.state is ControlState.REJECTED
    assert result.reason == "dry_run_backend_required"
