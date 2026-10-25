"""Torch helpers for rigid transforms and local xyz + rotation-vector poses.

Transforms use the convention ``T_A_B``: a point in frame B is mapped into
frame A by multiplication with ``T_A_B``. Inputs are assumed to be rigid
homogeneous transforms; no projection of malformed matrices is performed.
"""

from __future__ import annotations

import torch


def _check_tail(value: torch.Tensor, tail: tuple[int, ...], name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim < len(tail) or tuple(value.shape[-len(tail) :]) != tail:
        raise ValueError(f"{name} must have trailing shape {tail}")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must have a floating-point dtype")


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((zero, -z, y), dim=-1),
            torch.stack((z, zero, -x), dim=-1),
            torch.stack((-y, x, zero), dim=-1),
        ),
        dim=-2,
    )


def rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3) axis-angle vectors in radians to (..., 3, 3) matrices."""
    _check_tail(rotvec, (3,), "rotvec")
    theta2 = torch.sum(rotvec * rotvec, dim=-1, keepdim=True)
    # Clamp only the value used by the direct formulas. The Taylor branch below
    # retains smooth behavior at zero and avoids 0/0 in the unused branch.
    small_limit = 1e-3
    theta = torch.sqrt(theta2.clamp_min(small_limit**2))
    a_direct = torch.sin(theta) / theta
    b_direct = (1.0 - torch.cos(theta)) / theta.square()
    a_series = 1.0 - theta2 / 6.0 + theta2.square() / 120.0
    b_series = 0.5 - theta2 / 24.0 + theta2.square() / 720.0
    small = theta2 < small_limit**2
    a = torch.where(small, a_series, a_direct)[..., None]
    b = torch.where(small, b_series, b_direct)[..., None]

    skew = _skew(rotvec)
    identity = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    identity = identity.expand(*rotvec.shape[:-1], 3, 3)
    return identity + a * skew + b * (skew @ skew)


def matrix_to_rotvec(matrix: torch.Tensor) -> torch.Tensor:
    """Convert (..., 3, 3) rotation matrices to principal rotvecs.

    The largest quaternion component determines the extraction branch. This
    avoids dividing by a near-zero scalar component for rotations near pi.
    Matrices must represent proper rotations.
    """
    _check_tail(matrix, (3, 3), "matrix")
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((q_abs[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1].square(), m01 + m10, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m01 + m10, q_abs[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m02 + m20, m12 + m21, q_abs[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    denominators = (2.0 * q_abs).clamp_min(torch.finfo(matrix.dtype).tiny)
    candidates = candidates / denominators[..., :, None]
    choice = q_abs.argmax(dim=-1)
    gather_index = choice[..., None, None].expand(*choice.shape, 1, 4)
    quat = candidates.gather(dim=-2, index=gather_index).squeeze(dim=-2)
    quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True).clamp_min(
        torch.finfo(matrix.dtype).tiny
    )

    # q and -q encode the same rotation. Choosing nonnegative w gives the
    # principal angle in [0, pi], with a stable vector branch at pi.
    quat = torch.where(quat[..., :1] < 0, -quat, quat)
    scalar = quat[..., 0].clamp_min(0.0)
    vector = quat[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, scalar[..., None])
    eps = torch.finfo(matrix.dtype).eps
    scale_direct = angle / vector_norm.clamp_min(eps)
    scale_small = 2.0 / scalar[..., None].clamp_min(eps)
    scale = torch.where(vector_norm < 1e-6, scale_small, scale_direct)
    return vector * scale


def pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """Convert (..., 6) poses (xyz meters, then rotvec radians) to transforms."""
    _check_tail(pose, (6,), "pose")
    rotation = rotvec_to_matrix(pose[..., 3:])
    translation = pose[..., :3, None]
    upper = torch.cat((rotation, translation), dim=-1)
    bottom = torch.cat(
        (
            pose.new_zeros((*pose.shape[:-1], 1, 3)),
            pose.new_ones((*pose.shape[:-1], 1, 1)),
        ),
        dim=-1,
    )
    return torch.cat((upper, bottom), dim=-2)


def matrix_to_pose(transform: torch.Tensor) -> torch.Tensor:
    """Convert (..., 4, 4) transforms to local xyz + principal rotvec poses."""
    _check_tail(transform, (4, 4), "transform")
    return torch.cat((transform[..., :3, 3], matrix_to_rotvec(transform[..., :3, :3])), dim=-1)


def invert_transform(transform: torch.Tensor) -> torch.Tensor:
    """Invert rigid transforms (..., 4, 4) using Rᵀ and -Rᵀt."""
    _check_tail(transform, (4, 4), "transform")
    rotation_t = transform[..., :3, :3].transpose(-1, -2)
    translation = -(rotation_t @ transform[..., :3, 3, None])
    upper = torch.cat((rotation_t, translation), dim=-1)
    bottom = torch.cat(
        (
            transform.new_zeros((*transform.shape[:-2], 1, 3)),
            transform.new_ones((*transform.shape[:-2], 1, 1)),
        ),
        dim=-1,
    )
    return torch.cat((upper, bottom), dim=-2)


def relative_transform(reference: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """Return ``T_reference_pose = inverse(T_world_reference) @ T_world_pose``."""
    _check_tail(reference, (4, 4), "reference")
    _check_tail(pose, (4, 4), "pose")
    return invert_transform(reference) @ pose


def compose_gripper_target(
    target_pose: torch.Tensor,
    desired_object_in_target: torch.Tensor,
    object_in_gripper: torch.Tensor,
) -> torch.Tensor:
    """Compose the gripper goal from target and grasp transforms.

    With ``T_A_B`` denoting a transform from B into A, the result is
    ``T_frame_gripper = T_frame_target @ T_target_object @ inverse(T_gripper_object)``.
    Thus ``result @ object_in_gripper`` equals the desired object pose.
    """
    for name, value in (
        ("target_pose", target_pose),
        ("desired_object_in_target", desired_object_in_target),
        ("object_in_gripper", object_in_gripper),
    ):
        _check_tail(value, (4, 4), name)
    return target_pose @ desired_object_in_target @ invert_transform(object_in_gripper)
