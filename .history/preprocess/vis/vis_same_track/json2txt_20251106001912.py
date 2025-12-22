import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import json

def load_pose_from_json(file_path: str):
    """
    从 JSON 文件读取位姿，返回：
        poses_dict: {image_name: (R, t, euler_angles)}
        names_list: 图像名称列表
        origin_pose_dict: 原始 4x4 矩阵
    """
    coord_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])
    poses_dict = {}
    names_list = []
    origin_pose_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        camera_poses = json.load(f)

    for pose_data in camera_poses:
        name = pose_data["OriginalImageName"]
        T4x4_original = np.array(pose_data["T4x4"])
        T4x4_transformed = T4x4_original @ coord_transform
        R_c2w = T4x4_transformed[:3, :3]
        t_c2w = T4x4_transformed[:3, 3]
        euler_angles = R.from_matrix(R_c2w).as_euler('xyz', degrees=True)

        poses_dict[name] = (R_c2w, t_c2w, euler_angles)
        names_list.append(name)
        origin_pose_dict[name] = T4x4_original

    return poses_dict, names_list, origin_pose_dict

# ----------------------------
# 生成 txt
# ----------------------------
json_path = "/home/ubuntu/Documents/code/LXY/terra_ply_island/terra_ply/query/interval1_HKairport_GNSS02/sampleinfos_interpolated.json"
txt_save_path = "/home/ubuntu/Documents/code/LXY/terra_ply_island/terra_ply/query/interval1_HKairport_GNSS02/sampleinfos_interpolated.txt"

poses_dict, names_list, _ = load_pose_from_json(json_path)

with open(txt_save_path, "w", encoding="utf-8") as f:
    for name in names_list:
        _, t, euler = poses_dict[name]
        # t: x, y, z
        # euler: roll, pitch, yaw
        line = f"{name} {t[0]} {t[1]} {t[2]} {euler[0]} {euler[1]} {euler[2]}\n"
        f.write(line)

print(f"✅ 已生成 txt 文件：{txt_save_path}")
