# obj_lon, obj_lat, obj_alt = 112.991115, 28.295768,  25
import os
import json
import numpy as np
import json
import pandas as pd
from pathlib import Path
from wrapper import Camera
import pyproj
import random
from scipy.spatial.transform import Rotation as R
from wrapper import Pose, Camera
from transform import visualize_points_on_images
import cv2
import torch
from tqdm import tqdm
from torch import nn
from get_depth import sample_points_with_valid_depth, get_3D_samples, transform_ecef_origin, get_points2D_ECEF_projection
from transform import get_rotation_enu_in_ecef, WGS84_to_ECEF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 绘图支持

# ========= 配置 =========
image_root = "/mnt/sda/MapScape/query/images/feicuiwan_sim_seq7"
txt_path = "/mnt/sda/MapScape/query/bbox/feicuiwan_sim_seq7/coordinates.txt"
# output_root = "/mnt/sda/MapScape/sup/feicuiwan_sim_seq3/outputs"
# output_root = "/mnt/sda/MapScape/sup/object_index/outputs"
seq_name = "feicuiwan_sim_seq7"
output_root = os.path.join("/mnt/sda/MapScape/query/bbox" ,seq_name)
output_pose_path = os.path.join(output_root, seq_name +'.txt')
output_rtk_path = os.path.join(output_root, seq_name +'_RTK.txt')
output_bbox_path = os.path.join(output_root, seq_name +'_xy.txt')

os.makedirs(output_root, exist_ok=True)

# 相机参数
K = [1920, 1080, 1333.5, 1333.5, 960.0, 540.0]  # fx, fy, cx, cy
width, height = K[0], K[1]
cam_ref = {
    'model': 'PINHOLE',
    'width': width,
    'height': height,
    'params': [K[2], K[3], K[4], K[5]]
}
rcamera = Camera.from_colmap(cam_ref)

# ========== 读取数据 ==========
columns = [
    "FileName", "CameraLongitude", "CameraLatitude", "CameraHeight",
    "CameraRoll", "CameraPitch", "CameraYaw",
    "Target_class",
    "ObjectLongitude", "ObjectLatitude", "ObjectHeight",
    "ObjectRoll", "ObjectPitch", "ObjectYaw",
    "BBOXLength", "BBOXWidth", "BBOXHight"
]

# 读取时指定列名，并处理空格
# df = pd.read_csv(txt_path, delim_whitespace=True, header=None, names=columns, dtype={"FileName": str})
df = pd.read_csv(
    txt_path,
    delim_whitespace=True,
    skiprows=1,              # ← 跳过标题行
    header=None,
    names=columns,
    dtype={"FileName": str}
)
# df['FileIndex'] = df['FileName'].str.extract(r'(\d+)_0.png').astype(int)
df['FileIndex'] = df['FileName'].str.extract(r'(\d+)_0\.png').astype(int)
df = df.sort_values(by='FileIndex')

# 获取排序后的唯一 FileName 列表
sorted_names = df['FileName'].unique()

# 然后用 for 循环遍历这些 name，从原 df 中提取 group

# 按 FileIndex 排序
# df = df.sort_values(by='FileIndex')
# grouped = df.groupby('FileName')
num_cols = [
    "CameraLongitude", "CameraLatitude", "CameraHeight",
    "CameraRoll", "CameraPitch", "CameraYaw",
    "ObjectLongitude", "ObjectLatitude", "ObjectHeight",
    "ObjectRoll", "ObjectPitch", "ObjectYaw",
    "BBOXLength", "BBOXWidth", "BBOXHight"
]

df[num_cols] = df[num_cols].astype(float)
with open(output_pose_path, 'w') as f_pose, open(output_rtk_path, 'w') as f_rtk, open(output_bbox_path, 'w') as f_bbox:
    for name in sorted_names:
        group = df[df['FileName'] == name]
        image_path = os.path.join(image_root, name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"图像未找到: {image_path}")
            continue

        # 获取相机位姿（默认每一行都一致）
        row = group.iloc[0]
        cam_lon, cam_lat, cam_alt = row['CameraLongitude'], row['CameraLatitude'], row['CameraHeight']
        cam_roll, cam_pitch, cam_yaw = row['CameraRoll'], row['CameraPitch'], row['CameraYaw']
        R_pose = R.from_euler('xyz', [cam_pitch, cam_roll, cam_yaw], degrees=True).as_matrix()
        R_ned_to_ecef = get_rotation_enu_in_ecef(cam_lon, cam_lat)
        R_c2w = R_ned_to_ecef @ R_pose
        t_c2w = WGS84_to_ECEF([cam_lon, cam_lat, cam_alt])
        ref_T = np.eye(4)
        ref_T[:3, :3] = R_c2w
        ref_T[:3, 3] = t_c2w
        pose_ref_origin = transform_ecef_origin(ref_T, origin=np.zeros(3))

        # 遍历所有目标
        for idx, row in group.iterrows():
            obj_lon, obj_lat, obj_alt = row['ObjectLongitude'], row['ObjectLatitude'], row['ObjectHeight']
            obj_roll, obj_pitch, obj_yaw = row['ObjectRoll'], row['ObjectPitch'], row['ObjectYaw']
            target_class = row['Target_class']
            L, W, H = row['BBOXLength'], row['BBOXHight'], row['BBOXWidth']
            object_lens = np.array([L, H, W]) * 0.01  # 单位转换为 m

            # 用中心点投影（你可切换为使用8个角点）
            local_corners = np.array([[0, 0, 0]]).T  # 使用中心点

            R_obj_pose = R.from_euler('xyz', [obj_pitch, obj_roll, obj_yaw], degrees=True).as_matrix()
            R_obj_to_ecef = get_rotation_enu_in_ecef(obj_lon, obj_lat) @ R_obj_pose
            t_obj_ecef = np.array(WGS84_to_ECEF([obj_lon, obj_lat, obj_alt]))
            ref_T_obj = np.eye(4)
            ref_T_obj[:3, :3] = R_obj_to_ecef
            ref_T_obj[:3, 3] = t_obj_ecef
            pose_obj_origin = transform_ecef_origin(ref_T_obj, origin=np.zeros(3))

            corners_ecef = (t_obj_ecef.reshape(3, 1) + pose_obj_origin[:3, :3] @ local_corners).T
            pts2d, _, _, _ = get_points2D_ECEF_projection(pose_ref_origin, rcamera, corners_ecef, use_valid=False)
            if pts2d.shape[0] > 0:
                x, y = pts2d[0][:2].astype(int)
                if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                    cv2.circle(img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
                    cv2.putText(img, f'{row["Target_class"]}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                # 写入文件，格式为 "name lon lat alt roll pitch yaw"
            if x < 1920 and x >=0 and y < 1080 and y>=0:
                f_rtk.write(f"{name} {target_class} {obj_lon} {obj_lat} {obj_alt}\n")
                f_bbox.write(f"{name} {target_class} {x} {y}\n")
        f_pose.write(f"{name} {cam_lon} {cam_lat} {cam_alt} {cam_roll} {cam_pitch} {cam_yaw}\n")

        # 保存图像
        out_path = os.path.join(output_root, name)
        # cv2.imwrite(out_path, img)
        print(f"图像已保存: {out_path}")
