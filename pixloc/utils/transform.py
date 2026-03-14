"""Coordinate transformation and pose conversion utilities."""

import glob
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pyproj
import torch
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def move_inputs_to_cuda(*args, device: Optional[torch.device] = None):
    """Recursively move all Tensors, Pose, Camera objects to the target device.

    Returns a tuple in the same order as the input args.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def move(x):
        if isinstance(x, torch.Tensor):
            return x.to(device)
        elif hasattr(x, 'to') and callable(getattr(x, 'to')):
            return x.to(device)
        elif isinstance(x, (list, tuple)):
            return type(x)(move(i) for i in x)
        elif isinstance(x, dict):
            return {k: move(v) for k, v in x.items()}
        else:
            return x

    return tuple(move(arg) for arg in args)


# ---------------------------------------------------------------------------
# Coordinate system conversions (WGS84 <-> ECEF)
# ---------------------------------------------------------------------------

def WGS84_to_ECEF(pos: List[float]) -> List[float]:
    """Convert WGS84 (lon, lat, height) to ECEF (x, y, z)."""
    lon, lat, height = pos
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326",
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        always_xy=True,
    )
    x, y, z = transformer.transform(lon, lat, height, radians=False)
    return [x, y, z]


def ECEF_to_WGS84(pos: List[float]) -> List[float]:
    """Convert ECEF (x, y, z) to WGS84 (lon, lat, height)."""
    x, y, z = pos
    transformer = pyproj.Transformer.from_crs(
        {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'},
        "EPSG:4326",
        always_xy=True,
    )
    lon, lat, height = transformer.transform(x, y, z, radians=False)
    return [lon, lat, height]


def ECEF_to_WGS84_batch(ecef_coords: np.ndarray) -> np.ndarray:
    """Convert batch of ECEF coordinates to WGS84 (lon, lat, alt).

    Args:
        ecef_coords: Array of shape (n, 3) with columns [x, y, z].

    Returns:
        Array of shape (n, 3) with columns [lon_deg, lat_deg, alt_m].
    """
    x, y, z = ecef_coords[:, 0], ecef_coords[:, 1], ecef_coords[:, 2]

    a = 6378137.0
    e_sq = 6.69437999014e-3
    p = np.sqrt(x ** 2 + y ** 2)
    lon = np.arctan2(y, x)

    lat = np.arctan2(z, p * (1 - e_sq))
    N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
    alt = p / np.cos(lat) - N
    lat = np.arctan2(z * (N + alt), p * (N * (1 - e_sq) + alt))
    return np.stack([np.rad2deg(lon), np.rad2deg(lat), alt], axis=1)


# ---------------------------------------------------------------------------
# Rotation helpers (ENU <-> ECEF)
# ---------------------------------------------------------------------------

def get_rotation_enu_in_ecef(lon: float, lat: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from ENU to ECEF for given lon/lat (degrees).

    Reference:
        https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3
    """
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    up = np.array([
        np.cos(lon_rad) * np.cos(lat_rad),
        np.sin(lon_rad) * np.cos(lat_rad),
        np.sin(lat_rad),
    ])
    east = np.array([
        -np.sin(lon_rad),
        np.cos(lon_rad),
        0,
    ])
    north = np.cross(up, east)

    rot = np.zeros((3, 3))
    rot[:, 0] = east
    rot[:, 1] = north
    rot[:, 2] = up
    return rot


def get_rotation_enu_in_ecef_batch(
    lon: np.ndarray, lat: np.ndarray
) -> np.ndarray:
    """Batch computation of ENU-to-ECEF rotation matrices.

    Args:
        lon: (n,) longitude array in degrees.
        lat: (n,) latitude array in degrees.

    Returns:
        (n, 3, 3) rotation matrices.
    """
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    s_lon, c_lon = np.sin(lon_rad), np.cos(lon_rad)
    s_lat, c_lat = np.sin(lat_rad), np.cos(lat_rad)

    R_batch = np.zeros((len(lon), 3, 3))
    R_batch[:, 0, 0] = -s_lon
    R_batch[:, 1, 0] = c_lon
    R_batch[:, 2, 0] = 0
    R_batch[:, 0, 1] = -c_lon * s_lat
    R_batch[:, 1, 1] = -s_lon * s_lat
    R_batch[:, 2, 1] = c_lat
    R_batch[:, 0, 2] = c_lon * c_lat
    R_batch[:, 1, 2] = s_lon * c_lat
    R_batch[:, 2, 2] = s_lat
    return R_batch


# ---------------------------------------------------------------------------
# Euler / matrix conversions (numpy, ECEF frame)
# ---------------------------------------------------------------------------

def euler_angles_to_matrix_ECEF(
    euler_angles: List[float], trans: List[float]
) -> np.ndarray:
    """Convert Euler angles + WGS84 translation to a 4x4 c2w matrix in ECEF.

    Args:
        euler_angles: [pitch, roll, yaw] in degrees (ENU convention).
        trans: [lon, lat, alt] in WGS84.

    Returns:
        4x4 camera-to-world transformation matrix in ECEF.
    """
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)

    T_c2w = np.eye(4)
    T_c2w[:3, :3] = R_c2w
    T_c2w[:3, 3] = t_c2w
    return T_c2w


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
    ])


# ---------------------------------------------------------------------------
# Pose conversions (pixloc internal <-> OSG/WGS84)
# ---------------------------------------------------------------------------

def pixloc_to_osg_batch(T_batch_c2w: torch.Tensor) -> np.ndarray:
    """Convert a batch of c2w ECEF transforms to ENU Euler angles.

    Args:
        T_batch_c2w: (n, 4, 4) camera-to-world matrices in ECEF.

    Returns:
        (n, 3) Euler angles [pitch, roll, yaw] in ENU (degrees).
    """
    T_batch_c2w = T_batch_c2w.cpu().numpy()
    T_batch_c2w = np.asarray(T_batch_c2w)

    R_batch_c2w = T_batch_c2w[:, :3, :3]
    t_batch_c2w = T_batch_c2w[:, :3, 3]
    t_batch_wgs84 = ECEF_to_WGS84_batch(t_batch_c2w)
    lons, lats = t_batch_wgs84[:, 0], t_batch_wgs84[:, 1]
    rot_enu_in_ecef_batch = get_rotation_enu_in_ecef_batch(lons, lats)
    rot_ecef_in_enu_batch = np.transpose(rot_enu_in_ecef_batch, (0, 2, 1))
    rot_pose_in_enu_batch = rot_ecef_in_enu_batch @ R_batch_c2w
    rot_obj = R.from_matrix(rot_pose_in_enu_batch)
    euler_angles_batch = rot_obj.as_euler('xyz', degrees=True)
    return euler_angles_batch


def pixloc_to_osg(
    T_refined_c2w: np.ndarray,
) -> Tuple[np.ndarray, List[float], np.ndarray, np.ndarray]:
    """Convert a single c2w ECEF transform to ENU Euler angles and WGS84.

    Args:
        T_refined_c2w: 4x4 camera-to-world matrix in ECEF.

    Returns:
        Tuple of (euler_angles_enu, wgs84_position, T_ECEF_w2c, 6-dof pose).
    """
    R_c2w, t_c2w = T_refined_c2w[:3, :3], T_refined_c2w[:3, 3]
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w)
    lon, lat, _ = t_c2w_wgs84
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    rot_ecef_in_enu = rot_enu_in_ecef.T
    rot_pose_in_enu = np.matmul(rot_ecef_in_enu, R_c2w)
    rot_obj = R.from_matrix(rot_pose_in_enu)
    euler_angles_in_enu = rot_obj.as_euler('xyz', degrees=True)

    R_w2c = R_c2w.T
    t_w2c = np.array(-R_w2c.dot(t_c2w))
    T_ECEF = np.concatenate((R_w2c, np.array([t_w2c]).transpose()), axis=1)

    kf_pose = np.concatenate((t_c2w, euler_angles_in_enu))
    return euler_angles_in_enu, t_c2w_wgs84, T_ECEF, kf_pose


# ---------------------------------------------------------------------------
# Kalman filter predictor
# ---------------------------------------------------------------------------

def kf_predictor(
    observations: np.ndarray, num_candidates: int = 128
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """Predict the next pose using a Kalman filter on 6-DoF observations.

    The state vector is [x, y, z, pitch, roll, yaw, vx, vy, vz, vpitch, vroll, vyaw].

    Args:
        observations: (N, 6) array of past pose observations.
        num_candidates: (unused, kept for API compatibility).

    Returns:
        Tuple of (updated_observations, predicted_euler, predicted_wgs84).
    """
    observations = np.array(observations)
    assert observations.shape[1] == 6, "Expected input shape: (N, 6)"

    dt = 1.0
    A = np.eye(12)
    for i in range(6):
        A[i, i + 6] = dt

    H = np.hstack([np.eye(6), np.zeros((6, 6))])

    kf = KalmanFilter(
        transition_matrices=A,
        observation_matrices=H,
        initial_state_mean=np.zeros(12),
        initial_state_covariance=np.eye(12) * 0.1,
        transition_covariance=np.eye(12) * 0.05,
        observation_covariance=np.eye(6) * 0.1,
    )

    state_mean = np.zeros(12)
    state_mean[:6] = observations[0]
    state_cov = np.eye(12) * 0.1

    for obs in observations:
        state_mean, state_cov = kf.filter_update(
            state_mean, state_cov, observation=obs
        )

    next_state_mean = A @ state_mean
    most_likely_pose = next_state_mean[:6]

    euler = most_likely_pose[3:]
    t_c2w_ecef = most_likely_pose[:3]
    t_c2w_wgs84 = ECEF_to_WGS84(t_c2w_ecef)

    observations = np.vstack([observations, most_likely_pose])
    observations = observations[1:, :]

    return observations, euler, t_c2w_wgs84


# ---------------------------------------------------------------------------
# File / path utilities
# ---------------------------------------------------------------------------

def get_sorted_image_paths_uavscenes(
    directory_path: str, extension: str = ".jpg"
) -> List[str]:
    """Get sorted image paths from a UAVScenes directory.

    Files are sorted by the numeric timestamp in their filename.

    Args:
        directory_path: Directory containing image files.
        extension: File extension filter.

    Returns:
        List of full file paths, sorted by timestamp.
    """
    search_pattern = os.path.join(directory_path, f"*{extension}")
    image_paths = glob.glob(search_pattern)
    if not image_paths:
        logger.warning(
            "No '%s' files found in directory '%s'.", extension, directory_path
        )
        return []
    sorted_paths = sorted(
        image_paths,
        key=lambda path: float(os.path.splitext(os.path.basename(path))[0]),
    )
    return sorted_paths
