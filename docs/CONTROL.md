# Geometry and control handoff

This module provides rigid-transform utilities and a conservative control
handoff. It does not connect to a robot. The only included backend is
`DryRunBackend`, which records one bounded step in memory.

## Pose and frame convention

Model poses use a fixed local chart with six values:

```text
[x, y, z, rotvec_x, rotvec_y, rotvec_z]
```

Translation is in meters and the rotation vector is in radians. This chart is
not an SE(3) Haar measure, and these helpers make no claim about a density with
respect to Haar measure. Any normalized translation scale stored in config is
separate: the caller must explicitly convert and validate metric coordinates
before setting `metric_valid=True`. The geometry and control modules never
infer, apply, or undo that scale.

Transforms use `T_A_B` notation: `T_A_B @ point_B` expresses the point in A.
`relative_transform(T_world_reference, T_world_pose)` returns
`T_reference_pose`. `invert_transform` uses the rigid inverse `Rᵀ, -Rᵀt` and
assumes a valid homogeneous rigid transform. `rotvec_to_matrix` uses a Taylor
series around zero; `matrix_to_rotvec` extracts a principal rotation through a
largest-quaternion-component branch, including near pi.

For a target frame pose `T_frame_target`, desired object relation
`T_target_object`, and grasp relation `T_gripper_object`, the gripper goal is:

```text
T_frame_gripper = T_frame_target
                  @ T_target_object
                  @ inverse(T_gripper_object)
```

This ensures `T_frame_gripper @ T_gripper_object` equals the desired object
pose. `compose_gripper_target` implements this order.

## Plan gate

`RepairSpec` pins a request, object ID, target ID, reference version, frame,
validity interval, desired target transform, grasp transform, units, and
tolerances. `PlanReply` must echo the request and identities, use the same
reference version and frame, and provide a fresh timestamp, identity
confidence, calibrated status, metric-valid status, and posterior modes. The
default gate requires identity confidence at least 0.99 and posterior support
at least 0.95. It fails closed for uncalibrated output, unknown metric scale,
stale or future replies, mismatched IDs or versions, unsupported units, and
invalid posterior data.

Construct `prepare_bounded_step(spec, action_id=..., current_gripper_pose=...,
robot_state_version=..., held_object=...)` **before** requesting an outcome
posterior. This computes the gripper goal and applies the translation and
rotation limits once. `BoundedStep` contains the resulting start/end poses,
action ID, request/frame/reference IDs, robot state version, displacement
limits, path definition, and held-object attachment and geometry reference.

The predictor must condition on this exact bounded command and robot-state
snapshot. Its `PlanReply` must include `action_id`,
`command_fingerprint(command)`, `start_pose`, and `robot_state_version`.
The fingerprint is SHA-256 over canonical JSON of every command field,
including the held object's geometry ID/version and grasp transform. It
identifies an action; it is not a calibration or authenticity certificate.
The posterior producer is responsible for actually conditioning on that
action, rather than relabeling a posterior for a different motion.

`gate_plan` requires the actual `command`, `current_gripper_pose`, and
`robot_state_version` as keyword arguments. It checks the physical step bounds
and all reply bindings. Start poses must match exactly after conversion to
the documented chart values, and robot state versions must match exactly.
A changed start, state version, endpoint, attachment, or action requires a
new command and posterior. A posterior predicting successful full repair
cannot authorize a shorter command created afterward. If the actual bounded
action's posterior remains outside the repair tolerance, the gate rejects it.

Each `PosteriorMode` has an explicit probability and an `[S, 6]` array of
predicted resulting object poses **after the bound command**, in the request
frame. Samples receive a
success indicator only when both their Euclidean translation error and SO(3)
geodesic rotation error are within the requested tolerances. Posterior support
is the sum across modes of `mode_probability * within_tolerance_sample_fraction`.
No pose is averaged across samples or modes. Consequently, two incompatible
modes on opposite sides of the target cannot pass merely because their
arithmetic mean lies on the target. Mode probabilities must be finite,
nonnegative, and sum to one.

## State and backend boundary

The allowed lifecycle is:

```text
proposed -> accepted -> executing -> completed
     |          |             `-> aborted
     |          `-> rejected / expired
     `-> rejected / expired
```

Create `ControlSession(spec, reply, command)` with the already bounded command.
Both `accept(now_s, current_gripper_pose=..., robot_state_version=...)` and
`execute_one_step` require a fresh observed start pose and robot state version.
Acceptance saves the command fingerprint. Immediately before execution, the
session checks that fingerprint again and rechecks the full gate, including
the current start and state version. Replacing both the command and reply
after acceptance still fails. Execution never reconstructs or clips a
different command. The same immutable command is checked and passed to the
backend at most once; a failed backend result aborts the session.

`CollisionChecker.is_path_collision_free(command)` replaces the endpoint-only
interface. It must certify the **entire continuous swept path**, including
both endpoints, the robot links, gripper, and held object. The command path is
linear translation with rotation
`R(s) = R_start @ rotvec_to_matrix(s * matrix_to_rotvec(R_start.T @ R_end))`
for `s` in `[0, 1]`, using this package's principal rotvec branch. The checker
must resolve the robot configuration from `robot_state_version` and use the
same path realization as the backend. Unknown or unsupported robot state or
path realization must be rejected.

`HeldObject` specifies the object ID, versioned collision-geometry registry
key, and `T_gripper_object` attachment. The geometry is expressed in the
object's local frame and must conservatively enclose the held object. The
checker must use that geometry throughout the path and fail closed for
unknown/stale geometry. Endpoint clearance or a coarse list of sampled
clearances alone is insufficient: the checker must certify swept-volume
clearance, analytically or with conservative bounds between samples.
Exceptions, missing path-check methods, and non-`True` results reject the
command before backend submission. There is no endpoint-only fallback.

`ControlBackend` describes the one-step adapter interface for a future
explicitly supplied implementation. This repository contains no external
adapter, robot address, socket client, credential path, or hardware transport.
The current session requires `backend.is_dry_run is True` and refuses other
backends before submission. `DryRunBackend` only stores the bounded command
in memory. Its `completed`
state means the dry-run command was recorded; it does not mean a physical
action occurred. IK and collision checks are injected protocols so tests and
integrators can supply finite deterministic checks without adding a robot
connection here.

## Validation status

`tests/test_geometry_control.py` covers conversion stability, frame
composition, gate thresholds, multimodal target support, state transitions,
revalidation, bounded steps, full-repair versus truncated-action binding,
changed commands/start poses/robot versions, and a held object crossing a thin
obstacle despite both endpoints being clear. The implementation
subtask did not run tests or model code on the local machine; validation is
delegated to the root agent's HP workstation as requested. No hardware
validation has been performed or claimed.
