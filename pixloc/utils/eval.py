"""Evaluation utilities for pose and target location accuracy."""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from pixloc.utils.transform import WGS84_to_ECEF, get_rotation_enu_in_ecef

logger = logging.getLogger(__name__)


def _euler_to_rotation_ecef(
    euler_angles: list,
    trans: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert Euler angles and WGS84 translation to ECEF rotation and position.

    Args:
        euler_angles: [pitch, roll, yaw] in degrees.
        trans: [lon, lat, alt] in WGS84.

    Returns:
        (R_c2w, t_c2w) in ECEF coordinates.
    """
    lon, lat, _ = trans
    rot_pose_in_enu = R.from_euler(
        "xyz", euler_angles, degrees=True
    ).as_matrix()
    rot_enu_in_ecef = get_rotation_enu_in_ecef(lon, lat)
    R_c2w = np.matmul(rot_enu_in_ecef, rot_pose_in_enu)
    t_c2w = WGS84_to_ECEF(trans)
    return R_c2w, np.array(t_c2w)


def evaluate_pose(
    results_path: str,
    gt_path: str,
    only_localized: bool = False,
) -> Optional[str]:
    """Evaluate pose accuracy (translation and rotation) against ground truth.

    Args:
        results_path: Path to predicted poses file.
        gt_path: Path to ground-truth poses file.
        only_localized: If True, skip unlocalized images.

    Returns:
        Formatted evaluation string, or None on failure.
    """
    predictions: Dict[str, tuple] = {}
    test_names = []
    with open(results_path, "r") as f:
        for data in f.read().rstrip().split("\n"):
            tokens = data.split()
            name = tokens[0].split("/")[-1]
            t, e = np.split(np.array(tokens[1:], dtype=float), [3])
            e = [e[1], e[0], e[2]]
            R_c2w, t_c2w = _euler_to_rotation_ecef(e, t)
            predictions[name] = (R_c2w, t_c2w, e)
            test_names.append(name)

    gts: Dict[str, tuple] = {}
    with open(gt_path, "r") as f:
        for data in f.read().rstrip().split("\n"):
            tokens = data.split()
            name = tokens[0].split("/")[-1]
            t, e = np.split(np.array(tokens[1:], dtype=float), [3])
            e = [e[1], e[0], e[2]]
            R_c2w, t_c2w = _euler_to_rotation_ecef(e, t)
            gts[name] = (R_c2w, t_c2w, e)

    errors_t = []
    errors_R = []
    errors_yaw = []
    for name in test_names:
        if name not in predictions:
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.0
            e_yaw = 180.0
        else:
            R_gt, t_gt, euler_gt = gts[name]
            R_pred, t_pred, euler_pred = predictions[name]
            e_t = np.linalg.norm(-t_gt + t_pred, axis=0)
            cos = np.clip(
                (np.trace(np.dot(R_gt.T, R_pred)) - 1) / 2, -1.0, 1.0
            )
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            delta = (euler_pred[-1] - euler_gt[-1] + 180) % 360 - 180
            e_yaw = abs(delta)

        errors_t.append(e_t)
        errors_R.append(e_R)
        errors_yaw.append(e_yaw)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    errors_yaw = np.array(errors_yaw)

    med_t = np.median(errors_t)
    std_t = np.std(errors_t)
    med_R = np.median(errors_R)
    std_R = np.std(errors_R)
    med_yaw = np.median(errors_yaw)
    std_yaw = np.std(errors_yaw)

    out = "\nMedian errors: {:.3f}m, {:.3f}deg, {:.3f}deg\n".format(
        med_t, med_R, med_yaw
    )
    out += "Std errors: {:.3f}m, {:.3f}deg, {:.3f}deg\n".format(
        std_t, std_R, std_yaw
    )
    out += "Percentage of test images localized within:"
    threshs_t = [1, 3, 5]
    threshs_R = [1.0, 3.0, 5.0]
    for th_t in threshs_t:
        ratio = np.mean(errors_t < th_t)
        out += "\n\t{:.0f}cm: {:.2f}%".format(th_t * 100, ratio * 100)
    for th_R in threshs_R:
        ratio = np.mean(errors_R < th_R)
        out += "\n\t{:.1f}deg: {:.2f}%".format(th_R, ratio * 100)

    logger.info(out)
    return out


def evaluate_target(
    results_path: str,
    gt_path: str,
    only_localized: bool = False,
) -> Optional[Dict[str, float]]:
    """Evaluate target location accuracy in ECEF coordinates.

    Args:
        results_path: Path to predicted target locations file.
        gt_path: Path to ground-truth RTK target locations file.
        only_localized: If True, skip unlocalized images.

    Returns:
        Dictionary with error statistics, or None on failure.
    """
    if not os.path.exists(results_path):
        logger.warning("Prediction file not found: %s", results_path)
        return None
    if not os.path.exists(gt_path):
        logger.warning("Ground truth file not found: %s", gt_path)
        return None

    predictions: Dict[str, np.ndarray] = {}
    test_names = []
    total_num = 0
    test_num = 0

    with open(results_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split("/")[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                predictions[name] = t_ecef
                test_names.append(name)
                test_num += 1
            except Exception as e:
                logger.error("Failed to parse prediction for %s: %s", name, e)
                continue

    gts: Dict[str, np.ndarray] = {}
    with open(gt_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            name = tokens[0].split("/")[-1]
            try:
                t = np.array(tokens[1:4], dtype=float)
                t_ecef = WGS84_to_ECEF(t)
                gts[name] = t_ecef
                total_num += 1
            except Exception as e:
                logger.error("Failed to parse ground truth for %s: %s", name, e)
                continue

    errors_t = []
    for name in test_names:
        gt_name = name.split("_")[0] + "_0.png"
        if name not in predictions or gt_name not in gts:
            if only_localized:
                continue
            errors_t.append(np.inf)
        else:
            t_gt = np.array(gts[gt_name])
            t_pred = np.array(predictions[name])
            e_t = np.linalg.norm(t_pred - t_gt)
            errors_t.append(e_t)

    if len(errors_t) == 0:
        return None

    errors_t = np.array(errors_t)
    finite_errors = errors_t[np.isfinite(errors_t)]

    stats = {
        "MedianError": np.median(finite_errors),
        "StdError": np.std(finite_errors),
        "Recall@1m": np.mean(errors_t < 1),
        "Recall@3m": np.mean(errors_t < 3),
        "Recall@5m": np.mean(errors_t < 5),
        "Completeness": test_num / total_num if total_num > 0 else 0,
    }

    logger.info("Target evaluation: %s", stats)
    return stats
