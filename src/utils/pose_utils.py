"""Pose I/O utilities for loading poses and target points from files."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pixloc.pixlib.geometry import Pose
from pixloc.utils.transform import WGS84_to_ECEF, euler_angles_to_matrix_ECEF


def load_initial_pose(
    pose_file: str,
) -> Tuple[List[float], List[float], np.ndarray]:
    """Load the first valid pose from a pose file.

    Args:
        pose_file: Path to the pose file.

    Returns:
        (euler_angles [pitch, roll, yaw],
         translation  [lon, lat, alt],
         ecef_origin).

    Raises:
        ValueError: If no valid pose is found.
    """
    with open(pose_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1:])
                euler = [pitch, roll, yaw]
                trans = [lon, lat, alt]
                return euler, trans, WGS84_to_ECEF(trans)
    raise ValueError(f"No valid pose found in {pose_file}")


def load_pose_dict(
    pose_file: str,
    origin: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load all poses from a file and convert to PixLoc format.

    Args:
        pose_file: Path to the pose file.
        origin: ECEF origin for translation normalization.

    Returns:
        Dictionary mapping image names to pose entries containing
        ``T_w2c_4x4``, ``euler``, ``trans``, and ``T_w2c``.
    """
    pose_dict: Dict[str, Dict[str, Any]] = {}
    with open(pose_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            lon, lat, alt, roll, pitch, yaw = map(float, parts[1:])
            name = parts[0] if "_" in parts[0] else parts[0][:-4] + "_0.png"

            euler = [pitch, roll, yaw]
            trans = [lon, lat, alt]
            T_c2w = euler_angles_to_matrix_ECEF(euler, trans)

            entry: Dict[str, Any] = {"T_w2c_4x4": T_c2w.copy()}

            T_c2w[:3, 1] = -T_c2w[:3, 1]
            T_c2w[:3, 2] = -T_c2w[:3, 2]
            if origin is not None:
                T_c2w[:3, 3] -= origin

            T_w2c = np.eye(4)
            T_w2c[:3, :3] = T_c2w[:3, :3].T
            T_w2c[:3, 3] = -T_c2w[:3, :3].T @ T_c2w[:3, 3]

            entry["euler"] = euler
            entry["trans"] = trans
            entry["T_w2c"] = Pose.from_Rt(T_w2c[:3, :3], T_w2c[:3, 3]).to_flat()
            pose_dict[name] = entry
    return pose_dict


def load_target_points(xy_file: str) -> Dict[str, List[List[float]]]:
    """Load target 2D coordinates from a file.

    Args:
        xy_file: Path to the target coordinates file.

    Returns:
        Dictionary mapping image names to ``[[x, y]]`` coordinates.
    """
    xy_dict: Dict[str, List[List[float]]] = {}
    with open(xy_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                x, y = map(float, parts[1:])
                xy_dict[parts[0]] = [[x, y]]
    return xy_dict
