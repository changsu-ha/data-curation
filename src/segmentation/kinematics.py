"""Forward kinematics computation using Pinocchio.

Pinocchio is an optional dependency (conda-forge only, not pip-installable).
All functions in this module call ``require_pinocchio()`` at runtime and raise
a helpful ``ImportError`` if the library is absent, so importing this module
itself will always succeed.

Typical usage::

    from segmentation import kinematics
    from segmentation.robots.rby1 import EE_FRAMES

    model, data = kinematics.load_robot_model("/path/to/rby1.urdf")
    name_to_q_idx = kinematics.build_joint_index_map(model)
    positions, rotations, rpys = kinematics.compute_fk_trajectory(
        model, data, name_to_q_idx, joint_array, EE_FRAMES[0]
    )
    quaternions = kinematics.fk_to_quaternion(rotations)  # (N, 4) xyzw
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import pinocchio as pin

    _PINOCCHIO_AVAILABLE = True
except ImportError:  # pragma: no cover
    pin = None  # type: ignore[assignment]
    _PINOCCHIO_AVAILABLE = False


def require_pinocchio() -> None:
    """Raise ``ImportError`` with install instructions if pinocchio is absent."""
    if not _PINOCCHIO_AVAILABLE:
        raise ImportError(
            "pinocchio is required for forward kinematics but is not installed.\n"
            "Install it via conda:\n"
            "    conda install -c conda-forge pinocchio"
        )


# ---------------------------------------------------------------------------
# URDF / Pinocchio helpers
# ---------------------------------------------------------------------------

def load_robot_model(urdf_path: str) -> tuple[Any, Any]:
    """Load URDF and return ``(model, data)`` pinocchio objects.

    Parameters
    ----------
    urdf_path:
        Path to the robot URDF file.
    """
    require_pinocchio()
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    return model, data


def build_joint_index_map(model: Any) -> dict[str, int]:
    """Build a mapping from URDF joint name to index in the q vector.

    Pinocchio's ``model.joints[i].idx_q`` gives the start index in the
    configuration vector for joint ``i``. The universe joint (index 0) is
    skipped.

    Parameters
    ----------
    model:
        Pinocchio model returned by :func:`load_robot_model`.

    Returns
    -------
    dict mapping joint name → q-vector index.
    """
    require_pinocchio()
    name_to_q_idx: dict[str, int] = {}
    for i in range(1, model.njoints):  # skip universe joint at 0
        jname = model.names[i]
        name_to_q_idx[jname] = model.joints[i].idx_q
    return name_to_q_idx


def build_q_from_state(
    model: Any,
    name_to_q_idx: dict[str, int],
    joint_values: np.ndarray,
    joint_names: tuple[str, ...] | None = None,
    gripper_mapping: dict[str, int] | None = None,
) -> np.ndarray:
    """Map a dataset joint vector to Pinocchio's q vector.

    Parameters
    ----------
    model:
        Pinocchio model.
    name_to_q_idx:
        Joint name → q-vector index map from :func:`build_joint_index_map`.
    joint_values:
        1-D array of joint values from the dataset (e.g. 44-dim for RB-Y1).
    joint_names:
        Ordered joint names corresponding to the leading entries of
        ``joint_values``. Defaults to
        :data:`segmentation.robots.rby1.DATASET_JOINT_NAMES`.
    gripper_mapping:
        Dict mapping URDF gripper joint names to dataset vector indices.
        Defaults to :data:`segmentation.robots.rby1.GRIPPER_MAPPING`.

    Returns
    -------
    q : np.ndarray of shape ``(model.nq,)``
    """
    require_pinocchio()

    if joint_names is None:
        from .robots.rby1 import DATASET_JOINT_NAMES as _JN
        joint_names = _JN
    if gripper_mapping is None:
        from .robots.rby1 import GRIPPER_MAPPING as _GM
        gripper_mapping = _GM

    q = pin.neutral(model)

    for ds_idx, jname in enumerate(joint_names):
        if jname in name_to_q_idx:
            q[name_to_q_idx[jname]] = joint_values[ds_idx]

    for jname, ds_idx in gripper_mapping.items():
        if jname in name_to_q_idx and ds_idx < len(joint_values):
            q[name_to_q_idx[jname]] = joint_values[ds_idx]

    return q


def compute_fk(
    model: Any,
    data: Any,
    name_to_q_idx: dict[str, int],
    joint_values: np.ndarray,
    frame_name: str,
    joint_names: tuple[str, ...] | None = None,
    gripper_mapping: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute forward kinematics for a single timestep.

    Parameters
    ----------
    model, data:
        Pinocchio model and data objects.
    name_to_q_idx:
        Joint name → q-vector index map.
    joint_values:
        1-D joint state array from the dataset.
    frame_name:
        URDF frame name for the end-effector (e.g. ``"ee_right"``).
    joint_names, gripper_mapping:
        Robot-specific constants; see :func:`build_q_from_state`.

    Returns
    -------
    pos : np.ndarray (3,) — end-effector position [x, y, z] in metres.
    rot_matrix : np.ndarray (3, 3) — SO(3) rotation matrix.
    rpy : np.ndarray (3,) — roll, pitch, yaw in radians.
    """
    require_pinocchio()
    q = build_q_from_state(model, name_to_q_idx, joint_values, joint_names, gripper_mapping)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    frame_id = model.getFrameId(frame_name)
    oMf = data.oMf[frame_id]

    pos = oMf.translation.copy()
    rot_matrix = oMf.rotation.copy()
    rpy = pin.rpy.matrixToRpy(rot_matrix)

    return pos, rot_matrix, rpy


def compute_fk_trajectory(
    model: Any,
    data: Any,
    name_to_q_idx: dict[str, int],
    joint_array: np.ndarray,
    frame_name: str,
    joint_names: tuple[str, ...] | None = None,
    gripper_mapping: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute forward kinematics for an entire trajectory.

    Parameters
    ----------
    joint_array:
        2-D array of shape ``(N, D)`` where ``D`` is the dataset state
        dimensionality (e.g. 44 for RB-Y1).
    frame_name:
        URDF frame name for the end-effector.
    joint_names, gripper_mapping:
        Robot-specific constants; see :func:`build_q_from_state`.

    Returns
    -------
    positions : np.ndarray (N, 3)
    rot_matrices : np.ndarray (N, 3, 3)
    rpys : np.ndarray (N, 3)
    """
    require_pinocchio()
    n = len(joint_array)
    positions = np.empty((n, 3))
    rot_matrices = np.empty((n, 3, 3))
    rpys = np.empty((n, 3))

    for i in range(n):
        pos, rot, rpy = compute_fk(
            model, data, name_to_q_idx, joint_array[i], frame_name,
            joint_names, gripper_mapping,
        )
        positions[i] = pos
        rot_matrices[i] = rot
        rpys[i] = rpy

    return positions, rot_matrices, rpys


# ---------------------------------------------------------------------------
# Rotation representation bridge
# ---------------------------------------------------------------------------

def fk_to_quaternion(rot_matrices: np.ndarray) -> np.ndarray:
    """Convert an array of SO(3) rotation matrices to xyzw unit quaternions.

    This bridges the rotation matrix output of :func:`compute_fk_trajectory`
    to the ``[x, y, z, w]`` quaternion format expected by
    ``segmentation.features.build_features()`` (the ``cartesian[:, 3:7]``
    slice).

    Parameters
    ----------
    rot_matrices:
        Array of shape ``(N, 3, 3)`` containing SO(3) rotation matrices.

    Returns
    -------
    quaternions : np.ndarray of shape ``(N, 4)`` — columns ``[x, y, z, w]``.
    """
    n = len(rot_matrices)
    quaternions = np.empty((n, 4))

    for i, R in enumerate(rot_matrices):
        # Shepperd's method for numerical stability.
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        quaternions[i] = [x, y, z, w]

    return quaternions
