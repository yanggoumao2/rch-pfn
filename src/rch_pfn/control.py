"""Conservative, transport-free gate and one-step control handoff.

Poses are represented in the model's fixed local xyz + rotvec chart. At this
boundary xyz must already be metric meters and rotation must be radians; no
config normalization scale is inferred or applied here.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Protocol, Sequence

import torch

from .geometry import (
    compose_gripper_target,
    matrix_to_pose,
    matrix_to_rotvec,
    pose_to_matrix,
    relative_transform,
    rotvec_to_matrix,
)

Pose6 = tuple[float, float, float, float, float, float]


class ControlState(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ABORTED = "aborted"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass(frozen=True)
class RepairSpec:
    """Versioned target and tolerances for one repair request.

    ``target_pose`` is T_frame_target; ``desired_object_in_target`` is
    T_target_object; ``object_in_gripper`` is T_gripper_object. All three use
    the same xyz-meter + rotvec-radian pose chart.
    """

    request_id: str
    object_id: str
    target_id: str
    reference_version: int
    frame_id: str
    created_at_s: float
    expires_at_s: float
    target_pose: Pose6
    desired_object_in_target: Pose6
    object_in_gripper: Pose6
    position_unit: str = "m"
    rotation_unit: str = "rad"
    min_identity_confidence: float = 0.99
    min_posterior_support: float = 0.95
    translation_tolerance_m: float = 0.005
    rotation_tolerance_rad: float = 0.05
    max_reply_age_s: float = 0.5
    max_translation_step_m: float = 0.01
    max_rotation_step_rad: float = 0.1


@dataclass(frozen=True)
class PosteriorMode:
    """Weighted samples for one coherent predicted-result mode.

    ``poses`` has shape [samples, 6] and contains predicted resulting object
    poses after the exact bounded command bound by PlanReply, in the
    RepairSpec frame, in meters and radians.
    """

    mode_id: str
    probability: float
    poses: torch.Tensor | Sequence[Sequence[float]]


@dataclass(frozen=True)
class PlanReply:
    """Posterior conditioned on one already constructed, immutable command."""

    request_id: str
    object_id: str
    target_id: str
    reference_version: int
    frame_id: str
    generated_at_s: float
    identity_confidence: float
    calibrated: bool
    metric_valid: bool
    posterior_modes: tuple[PosteriorMode, ...]
    action_id: str
    command_fingerprint: str
    start_pose: Pose6
    robot_state_version: int
    position_unit: str = "m"
    rotation_unit: str = "rad"


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    posterior_support: float = 0.0
    selected_mode_id: str | None = None
    mode_supports: tuple[tuple[str, float], ...] = ()
    gripper_target_pose: Pose6 | None = None

    @property
    def rejected_reason(self) -> str | None:
        return None if self.accepted else self.reason


@dataclass(frozen=True)
class HeldObject:
    """Collision shape registry reference and rigid attachment to the gripper.

    The checker must resolve geometry_id + geometry_version to a conservative
    shape in the object's local frame. Unknown geometry must fail closed.
    """

    object_id: str
    geometry_id: str
    geometry_version: int
    object_in_gripper: Pose6


@dataclass(frozen=True)
class BoundedStep:
    request_id: str
    frame_id: str
    action_id: str
    reference_version: int
    robot_state_version: int
    start_pose: Pose6
    target_pose: Pose6
    translation_step_m: float
    rotation_step_rad: float
    max_translation_step_m: float
    max_rotation_step_rad: float
    held_object: HeldObject
    path_type: str = "linear_translation_so3_geodesic"


def command_fingerprint(command: BoundedStep) -> str:
    """SHA-256 identity of every command field, including path and attachment."""
    payload = {"format": "bounded-step-v1", "command": asdict(command)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BackendStepResult:
    success: bool
    detail: str
    dry_run: bool


@dataclass(frozen=True)
class ExecutionResult:
    state: ControlState
    reason: str
    command: BoundedStep | None = None
    backend_result: BackendStepResult | None = None


class IKChecker(Protocol):
    """Injected finite, deterministic reachability check; no robot connection."""

    def is_reachable(self, pose: Pose6, *, frame_id: str) -> bool: ...


class CollisionChecker(Protocol):
    """Check the entire swept path, including robot links and held object.

    For s in [0, 1], translation is linearly interpolated and rotation follows
    the shortest SO(3) geodesic from command.start_pose to command.target_pose.
    Its axis is this package's matrix_to_rotvec(R_start.T @ R_end) branch;
    use R(s) = R_start @ rotvec_to_matrix(s * axis).
    The checker must resolve the robot configuration from robot_state_version
    and the held-object geometry from its versioned registry reference. It
    must certify the continuous path (including endpoints), or return False.
    Endpoint checks or samples without conservative intersample bounds do not
    certify a path. The backend must use the same path realization.
    Unknown geometry, state, or unsupported path must fail closed.
    """

    def is_path_collision_free(self, command: BoundedStep) -> bool: ...


class ControlBackend(Protocol):
    """Record the exact checked command, with no reclipping or path substitution.

    This package provides no network adapter; ControlSession permits dry runs
    only. Future adapters must preserve the fingerprinted path semantics.
    """

    @property
    def is_dry_run(self) -> bool: ...

    def execute_step(self, command: BoundedStep) -> BackendStepResult: ...


class DryRunBackend:
    """Explicit local recorder. It never opens a socket or controls hardware."""

    is_dry_run = True

    def __init__(self) -> None:
        self.commands: list[BoundedStep] = []

    def execute_step(self, command: BoundedStep) -> BackendStepResult:
        self.commands.append(command)
        return BackendStepResult(
            success=True,
            detail="one bounded step recorded by DryRunBackend; no hardware was contacted",
            dry_run=True,
        )


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _pose_tensor(value: object) -> torch.Tensor | None:
    try:
        if isinstance(value, torch.Tensor):
            pose = value.detach().to(device="cpu", dtype=torch.float64)
        else:
            pose = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    except (TypeError, ValueError, RuntimeError):
        return None
    if tuple(pose.shape) != (6,) or not bool(torch.isfinite(pose).all()):
        return None
    return pose


def _identity_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _version_valid(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _pose_values(value: object) -> Pose6 | None:
    pose = _pose_tensor(value)
    return None if pose is None else tuple(float(item) for item in pose.tolist())


def _validate_held_object(spec: RepairSpec, held: HeldObject) -> str | None:
    if not isinstance(held, HeldObject):
        return "missing_held_object"
    if held.object_id != spec.object_id:
        return "held_object_identity_mismatch"
    if not _identity_string(held.geometry_id) or not _version_valid(held.geometry_version):
        return "invalid_held_object_geometry"
    attachment = _pose_values(held.object_in_gripper)
    if attachment is None or attachment != _pose_values(spec.object_in_gripper):
        return "held_object_attachment_mismatch"
    return None


def _validate_command(spec: RepairSpec, command: BoundedStep) -> str | None:
    if not isinstance(command, BoundedStep):
        return "invalid_bounded_step"
    if not _identity_string(command.action_id):
        return "invalid_action_id"
    if command.request_id != spec.request_id or command.frame_id != spec.frame_id:
        return "command_request_or_frame_mismatch"
    if not _version_valid(command.reference_version):
        return "invalid_command_reference_version"
    if command.reference_version != spec.reference_version:
        return "command_reference_version_mismatch"
    if not _version_valid(command.robot_state_version):
        return "invalid_robot_state_version"
    held_error = _validate_held_object(spec, command.held_object)
    if held_error is not None:
        return held_error
    # Commands contain immutable Python values, not mutable tensor/list views.
    for pose in (command.start_pose, command.target_pose, command.held_object.object_in_gripper):
        if not isinstance(pose, tuple) or len(pose) != 6 or not all(map(_finite_number, pose)):
            return "invalid_command_pose"
    if command.path_type != "linear_translation_so3_geodesic":
        return "unsupported_command_path"
    for value in (
        command.translation_step_m,
        command.rotation_step_rad,
        command.max_translation_step_m,
        command.max_rotation_step_rad,
    ):
        if not _finite_number(value) or value < 0.0:
            return "invalid_command_bounds"
    if (
        command.max_translation_step_m != spec.max_translation_step_m
        or command.max_rotation_step_rad != spec.max_rotation_step_rad
    ):
        return "command_limits_mismatch"
    error_pose = matrix_to_pose(
        relative_transform(
            pose_to_matrix(_pose_tensor(command.start_pose)),
            pose_to_matrix(_pose_tensor(command.target_pose)),
        )
    )
    distance = float(torch.linalg.vector_norm(error_pose[:3]))
    angle = float(torch.linalg.vector_norm(error_pose[3:]))
    if not math.isfinite(distance) or not math.isfinite(angle):
        return "invalid_command_pose"
    if distance > spec.max_translation_step_m + 1e-9 or angle > spec.max_rotation_step_rad + 1e-9:
        return "command_exceeds_bounds"
    if not math.isclose(
        distance, command.translation_step_m, rel_tol=1e-8, abs_tol=1e-9
    ) or not math.isclose(angle, command.rotation_step_rad, rel_tol=1e-8, abs_tol=1e-9):
        return "command_displacement_mismatch"
    return None


def _validate_spec(spec: RepairSpec) -> str | None:
    for name in ("request_id", "object_id", "target_id", "frame_id"):
        if not _identity_string(getattr(spec, name, None)):
            return f"invalid_{name}"
    if (
        not isinstance(spec.reference_version, int)
        or isinstance(spec.reference_version, bool)
        or spec.reference_version < 0
    ):
        return "invalid_reference_version"
    if spec.position_unit != "m" or spec.rotation_unit != "rad":
        return "unsupported_units"
    for name in ("created_at_s", "expires_at_s"):
        if not _finite_number(getattr(spec, name, None)):
            return "invalid_spec_time"
    if spec.expires_at_s <= spec.created_at_s:
        return "invalid_validity_interval"
    for name in ("target_pose", "desired_object_in_target", "object_in_gripper"):
        if _pose_tensor(getattr(spec, name, None)) is None:
            return f"invalid_{name}"
    for name in (
        "min_identity_confidence",
        "min_posterior_support",
        "translation_tolerance_m",
        "rotation_tolerance_rad",
        "max_reply_age_s",
        "max_translation_step_m",
        "max_rotation_step_rad",
    ):
        value = getattr(spec, name, None)
        if not _finite_number(value):
            return f"invalid_{name}"
    if not 0.0 <= spec.min_identity_confidence <= 1.0:
        return "invalid_min_identity_confidence"
    if not 0.0 < spec.min_posterior_support <= 1.0:
        return "invalid_min_posterior_support"
    if (
        min(
            spec.translation_tolerance_m,
            spec.rotation_tolerance_rad,
            spec.max_reply_age_s,
            spec.max_translation_step_m,
            spec.max_rotation_step_rad,
        )
        <= 0.0
    ):
        return "invalid_nonpositive_tolerance_or_step"
    return None


def gate_plan(
    spec: RepairSpec,
    reply: PlanReply,
    now_s: float,
    *,
    command: BoundedStep,
    current_gripper_pose: object,
    robot_state_version: int,
) -> GateDecision:
    """Fail-closed request gate using posterior mass inside target tolerances.

    Each sample is checked independently against the desired object transform.
    In-tolerance sample mass is summed across modes; poses are never averaged.
    """
    spec_error = _validate_spec(spec)
    if spec_error is not None:
        return GateDecision(False, spec_error)
    if not _finite_number(now_s):
        return GateDecision(False, "invalid_current_time")
    if now_s < spec.created_at_s:
        return GateDecision(False, "request_not_yet_valid")
    if now_s >= spec.expires_at_s:
        return GateDecision(False, "expired")

    if not isinstance(reply, PlanReply):
        return GateDecision(False, "invalid_plan_reply")
    for name in ("request_id", "object_id", "target_id", "frame_id"):
        if not _identity_string(getattr(reply, name, None)):
            return GateDecision(False, f"invalid_reply_{name}")
    if (
        not isinstance(reply.reference_version, int)
        or isinstance(reply.reference_version, bool)
        or reply.reference_version < 0
    ):
        return GateDecision(False, "invalid_reply_reference_version")
    if reply.request_id != spec.request_id:
        return GateDecision(False, "request_id_mismatch")
    if reply.object_id != spec.object_id:
        return GateDecision(False, "object_identity_mismatch")
    if reply.target_id != spec.target_id:
        return GateDecision(False, "target_identity_mismatch")
    if reply.reference_version != spec.reference_version:
        return GateDecision(False, "reference_version_mismatch")
    if reply.frame_id != spec.frame_id:
        return GateDecision(False, "frame_mismatch")
    if reply.position_unit != "m" or reply.rotation_unit != "rad":
        return GateDecision(False, "unsupported_units")

    command_error = _validate_command(spec, command)
    if command_error is not None:
        return GateDecision(False, command_error)
    if not _identity_string(reply.action_id) or reply.action_id != command.action_id:
        return GateDecision(False, "action_id_mismatch")
    if reply.command_fingerprint != command_fingerprint(command):
        return GateDecision(False, "command_fingerprint_mismatch")
    if _pose_values(reply.start_pose) != command.start_pose:
        return GateDecision(False, "reply_start_pose_mismatch")
    current_pose = _pose_values(current_gripper_pose)
    if current_pose is None:
        return GateDecision(False, "invalid_current_gripper_pose")
    if current_pose != command.start_pose:
        return GateDecision(False, "current_start_pose_mismatch")
    if (
        not _version_valid(robot_state_version)
        or not _version_valid(reply.robot_state_version)
        or reply.robot_state_version != command.robot_state_version
        or robot_state_version != command.robot_state_version
    ):
        return GateDecision(False, "robot_state_version_mismatch")

    if not _finite_number(reply.generated_at_s):
        return GateDecision(False, "invalid_reply_time")
    if reply.generated_at_s < spec.created_at_s:
        return GateDecision(False, "reply_predates_request")
    if reply.generated_at_s > now_s:
        return GateDecision(False, "reply_from_future")
    if now_s - reply.generated_at_s > spec.max_reply_age_s:
        return GateDecision(False, "stale_reply")

    if reply.calibrated is not True:
        return GateDecision(False, "uncalibrated")
    if reply.metric_valid is not True:
        return GateDecision(False, "metric_scale_invalid")
    if not _finite_number(reply.identity_confidence) or not 0.0 <= reply.identity_confidence <= 1.0:
        return GateDecision(False, "invalid_identity_confidence")
    if reply.identity_confidence < spec.min_identity_confidence:
        return GateDecision(False, "identity_confidence_below_threshold")

    modes = reply.posterior_modes
    if not isinstance(modes, (tuple, list)) or not modes:
        return GateDecision(False, "missing_posterior_modes")
    seen_mode_ids: set[str] = set()
    weights: list[float] = []
    parsed_modes: list[tuple[str, float, torch.Tensor]] = []
    for mode in modes:
        if not isinstance(mode, PosteriorMode) or not _identity_string(mode.mode_id):
            return GateDecision(False, "invalid_posterior_mode")
        if mode.mode_id in seen_mode_ids:
            return GateDecision(False, "duplicate_mode_id")
        seen_mode_ids.add(mode.mode_id)
        if not _finite_number(mode.probability) or mode.probability < 0.0:
            return GateDecision(False, "invalid_mode_probability")
        try:
            samples = (
                mode.poses.detach().to(device="cpu", dtype=torch.float64)
                if isinstance(mode.poses, torch.Tensor)
                else torch.as_tensor(mode.poses, dtype=torch.float64, device="cpu")
            )
        except (TypeError, ValueError, RuntimeError):
            return GateDecision(False, "invalid_posterior_samples")
        if samples.ndim != 2 or samples.shape[0] < 1 or samples.shape[1] != 6:
            return GateDecision(False, "invalid_posterior_samples")
        if not bool(torch.isfinite(samples).all()):
            return GateDecision(False, "invalid_posterior_samples")
        probability = float(mode.probability)
        weights.append(probability)
        parsed_modes.append((mode.mode_id, probability, samples))
    if abs(sum(weights) - 1.0) > 1e-6:
        return GateDecision(False, "mode_probabilities_not_normalized")

    target = pose_to_matrix(_pose_tensor(spec.target_pose))
    desired = pose_to_matrix(_pose_tensor(spec.desired_object_in_target))
    goal_object = target @ desired
    mode_supports: list[tuple[str, float]] = []
    total_support = 0.0
    for mode_id, probability, samples in parsed_modes:
        sample_transforms = pose_to_matrix(samples)
        errors = matrix_to_pose(
            relative_transform(goal_object.expand_as(sample_transforms), sample_transforms)
        )
        position_error = torch.linalg.vector_norm(errors[..., :3], dim=-1)
        rotation_error = torch.linalg.vector_norm(errors[..., 3:], dim=-1)
        in_tolerance = (position_error <= spec.translation_tolerance_m) & (
            rotation_error <= spec.rotation_tolerance_rad
        )
        support = probability * float(in_tolerance.to(torch.float64).mean())
        total_support += support
        mode_supports.append((mode_id, support))

    best_mode_id, _ = max(mode_supports, key=lambda item: item[1])
    if total_support + 1e-12 < spec.min_posterior_support:
        return GateDecision(
            False,
            "insufficient_posterior_support",
            posterior_support=total_support,
            selected_mode_id=best_mode_id,
            mode_supports=tuple(mode_supports),
        )

    return GateDecision(
        True,
        "accepted",
        posterior_support=total_support,
        selected_mode_id=best_mode_id,
        mode_supports=tuple(mode_supports),
        gripper_target_pose=command.target_pose,
    )


_ALLOWED_TRANSITIONS: dict[ControlState, frozenset[ControlState]] = {
    ControlState.PROPOSED: frozenset(
        (ControlState.ACCEPTED, ControlState.REJECTED, ControlState.EXPIRED)
    ),
    ControlState.ACCEPTED: frozenset(
        (ControlState.EXECUTING, ControlState.REJECTED, ControlState.EXPIRED)
    ),
    ControlState.EXECUTING: frozenset((ControlState.COMPLETED, ControlState.ABORTED)),
    ControlState.COMPLETED: frozenset(),
    ControlState.ABORTED: frozenset(),
    ControlState.REJECTED: frozenset(),
    ControlState.EXPIRED: frozenset(),
}


def prepare_bounded_step(
    spec: RepairSpec,
    *,
    action_id: str,
    current_gripper_pose: object,
    robot_state_version: int,
    held_object: HeldObject,
) -> BoundedStep:
    """Construct the exact candidate before asking for its outcome posterior.

    The resulting start/end, interpolation, robot state version and attached
    shape are immutable. Clipping happens here, never after posterior approval.
    """
    spec_error = _validate_spec(spec)
    if spec_error is not None:
        raise ValueError(spec_error)
    if not _identity_string(action_id) or not _version_valid(robot_state_version):
        raise ValueError("invalid_action_id_or_robot_state_version")
    held_error = _validate_held_object(spec, held_object)
    if held_error is not None:
        raise ValueError(held_error)
    current = _pose_tensor(current_gripper_pose)
    if current is None:
        raise ValueError("invalid_current_gripper_pose")
    current_transform = pose_to_matrix(current)
    target_transform = compose_gripper_target(
        pose_to_matrix(_pose_tensor(spec.target_pose)),
        pose_to_matrix(_pose_tensor(spec.desired_object_in_target)),
        pose_to_matrix(_pose_tensor(spec.object_in_gripper)),
    )
    delta_translation = target_transform[:3, 3] - current_transform[:3, 3]
    distance = float(torch.linalg.vector_norm(delta_translation))
    if distance > spec.max_translation_step_m:
        delta_translation = delta_translation * (spec.max_translation_step_m / distance)

    relative_rotation = relative_transform(current_transform, target_transform)[:3, :3]
    # Rotation error is the rotvec of R_current^T R_target.
    delta_rotation = matrix_to_rotvec(relative_rotation)
    angle = float(torch.linalg.vector_norm(delta_rotation))
    if angle > spec.max_rotation_step_rad:
        delta_rotation = delta_rotation * (spec.max_rotation_step_rad / angle)
    step_rotation = current_transform[:3, :3] @ rotvec_to_matrix(delta_rotation)

    step_transform = current_transform.clone()
    step_transform[:3, :3] = step_rotation
    step_transform[:3, 3] = current_transform[:3, 3] + delta_translation
    step_pose = matrix_to_pose(step_transform)
    command = BoundedStep(
        request_id=spec.request_id,
        frame_id=spec.frame_id,
        action_id=action_id,
        reference_version=spec.reference_version,
        robot_state_version=robot_state_version,
        start_pose=tuple(float(value) for value in current.tolist()),
        target_pose=tuple(float(value) for value in step_pose.tolist()),
        translation_step_m=float(torch.linalg.vector_norm(delta_translation)),
        rotation_step_rad=min(angle, spec.max_rotation_step_rad),
        max_translation_step_m=spec.max_translation_step_m,
        max_rotation_step_rad=spec.max_rotation_step_rad,
        held_object=replace(
            held_object, object_in_gripper=_pose_values(held_object.object_in_gripper)
        ),
    )
    command_error = _validate_command(spec, command)
    if command_error is not None:
        raise ValueError(command_error)
    return command


@dataclass
class ControlSession:
    """Strict single-proposal state machine with a revalidated one-step handoff."""

    spec: RepairSpec
    reply: PlanReply
    command: BoundedStep
    state: ControlState = field(default=ControlState.PROPOSED, init=False)
    decision: GateDecision | None = field(default=None, init=False)
    rejected_reason: str | None = field(default=None, init=False)
    _accepted_fingerprint: str | None = field(default=None, init=False, repr=False)

    def _transition(self, next_state: ControlState, reason: str | None = None) -> None:
        if next_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise RuntimeError(
                f"invalid control transition: {self.state.value} -> {next_state.value}"
            )
        self.state = next_state
        if next_state in (ControlState.REJECTED, ControlState.EXPIRED, ControlState.ABORTED):
            self.rejected_reason = reason or next_state.value

    def accept(
        self, now_s: float, *, current_gripper_pose: object, robot_state_version: int
    ) -> GateDecision:
        if self.state is not ControlState.PROPOSED:
            raise RuntimeError("only a proposed request can be accepted")
        self.decision = gate_plan(
            self.spec,
            self.reply,
            now_s,
            command=self.command,
            current_gripper_pose=current_gripper_pose,
            robot_state_version=robot_state_version,
        )
        if self.decision.accepted:
            self._accepted_fingerprint = command_fingerprint(self.command)
            self._transition(ControlState.ACCEPTED)
        elif self.decision.reason == "expired":
            self._transition(ControlState.EXPIRED, self.decision.reason)
        else:
            self._transition(ControlState.REJECTED, self.decision.reason)
        return self.decision

    def execute_one_step(
        self,
        *,
        now_s: float,
        current_gripper_pose: object,
        robot_state_version: int,
        backend: ControlBackend,
        ik: IKChecker,
        collision: CollisionChecker,
    ) -> ExecutionResult:
        if self.state is not ControlState.ACCEPTED:
            raise RuntimeError("only an accepted request can enter execution")

        # Capture one immutable command for the gate, path check and backend.
        # Even swapping both command and reply after acceptance cannot rebind it.
        command = self.command
        try:
            unchanged = command_fingerprint(command) == self._accepted_fingerprint
        except (TypeError, ValueError):
            unchanged = False
        if not unchanged:
            self._transition(ControlState.REJECTED, "accepted_command_changed")
            return ExecutionResult(self.state, "accepted_command_changed")
        self.decision = gate_plan(
            self.spec,
            self.reply,
            now_s,
            command=command,
            current_gripper_pose=current_gripper_pose,
            robot_state_version=robot_state_version,
        )
        if not self.decision.accepted:
            next_state = (
                ControlState.EXPIRED if self.decision.reason == "expired" else ControlState.REJECTED
            )
            self._transition(next_state, self.decision.reason)
            return ExecutionResult(self.state, self.decision.reason)

        if getattr(backend, "is_dry_run", None) is not True:
            self._transition(ControlState.REJECTED, "dry_run_backend_required")
            return ExecutionResult(self.state, "dry_run_backend_required", command)

        try:
            if not ik.is_reachable(command.target_pose, frame_id=command.frame_id):
                self._transition(ControlState.REJECTED, "ik_unreachable")
                return ExecutionResult(
                    self.state, self.rejected_reason or "ik_unreachable", command
                )
        except Exception:
            self._transition(ControlState.REJECTED, "ik_check_failed")
            return ExecutionResult(self.state, self.rejected_reason or "ik_check_failed", command)
        try:
            if collision.is_path_collision_free(command) is not True:
                self._transition(ControlState.REJECTED, "collision_check_failed")
                return ExecutionResult(
                    self.state, self.rejected_reason or "collision_check_failed", command
                )
        except Exception:
            self._transition(ControlState.REJECTED, "collision_check_error")
            return ExecutionResult(
                self.state, self.rejected_reason or "collision_check_error", command
            )

        self._transition(ControlState.EXECUTING)
        try:
            backend_result = backend.execute_step(command)
            if not isinstance(backend_result, BackendStepResult):
                raise TypeError("backend must return BackendStepResult")
            if backend_result.dry_run is not True:
                raise ValueError("backend must report a dry run")
        except Exception:
            self._transition(ControlState.ABORTED, "backend_step_failed")
            return ExecutionResult(
                self.state, self.rejected_reason or "backend_step_failed", command
            )

        if backend_result.success is True:
            self._transition(ControlState.COMPLETED)
            reason = "dry_run_recorded"
        else:
            self._transition(ControlState.ABORTED, "backend_step_failed")
            reason = self.rejected_reason or "backend_step_failed"
        return ExecutionResult(self.state, reason, command, backend_result)
