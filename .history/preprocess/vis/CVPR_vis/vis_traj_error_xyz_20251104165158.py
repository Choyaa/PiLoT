import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from transform import wgs84tocgcs2000_batch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# (新修改) 导入 Savitzky-Golay 滤波器
from scipy.signal import savgol_filter
# --- 以下代码保持严格不变 ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelweight'] = 'bold'
methods_name = {
    "GT": "GT",
    "FPVLoc": "PiLoT (Ours)",
    "Pixloc": "PixLoc",
    "Render2loc": "Render2Loc",
    "ORB@per30": "Render2ORB",
    "Render2loc@raft": "Render2RAFT"
}
methods = {
    "GT": "black",
    "FPVLoc": "#007F49",      # ✅ GeoPixel：深绿，不变
    "Pixloc": "#86AED5",       # 加深灰蓝
    "Render2loc": "#EF6C5D",   # 加深橘粉
    "ORB@per30": "#C79ACD",    # 加深淡紫
    "Render2loc@raft": "#F7B84A"  # 奶油橙
}
offset_dict = {
    "GT": (0.0, 0.0),
    "FPVLoc": (0.0, 0.0),
    "Pixloc": (-1.0, -1.0),
    "Render2loc": (1.0, -0.5),
    "ORB@per30": (-1.0, 1.0),
    "Render2loc@raft": (1.0, 1.0)
}
data_root = "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)
from matplotlib.patches import Circle
def plot_trajectory_with_error_circle(ax, traj, pos_err, color, linewidth=2.5, alpha=0.4, scale=0.8):
    """
    在轨迹上绘制误差圆圈，透明度和大小随误差变化
    - traj: (N, 2)
    - pos_err: (N,), 单位: m
    """
    x, y = traj[:, 0], traj[:, 1]
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(pos_err[i]) or pos_err[i] <= 0:
            continue
        radius = pos_err[i] * scale  # 缩放比例调节圆大小
        circle = Circle((x[i], y[i]), radius=radius,
                        color=color, alpha=alpha, linewidth=0)
        ax.add_patch(circle)
    # 绘制主轨迹线（在误差圈上方）
    valid = ~np.isnan(x)
    ax.plot(x[valid], y[valid], color=color, linewidth=linewidth, zorder=10)
def load_pose_with_name(file_path):
    data, angles, timestamps = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                name = parts[0]
                if "_" in name:
                    frame_idx = int(name.split("_")[0])
                else:
                    continue
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                if yaw < 0: yaw += 360
                data.append((lon, lat, alt))
                angles.append((pitch, yaw))
                timestamps.append(frame_idx)
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(timestamps), np.array(xyz), np.array(angles)
def load_pose(file_path):
    data, angles = [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                lon, lat, alt = map(float, parts[1:4])
                roll, pitch, yaw = map(float, parts[4:7])
                if yaw < 0: yaw += 360
                data.append((lon, lat, alt))
                angles.append((pitch, yaw))
    xyz = wgs84tocgcs2000_batch(data, 4547)
    return np.array(xyz), np.array(angles)
def align_to_full_timestamps(full_timestamps, timestamps_est, values):
    aligned = np.full((len(full_timestamps), values.shape[1]), np.nan, dtype=np.float32)
    idx_map = {t: i for i, t in enumerate(full_timestamps)}
    for i, t in enumerate(timestamps_est):
        if t in idx_map:
            aligned[idx_map[t]] = values[i]
    return aligned
def transform_points(points, scale, R, t):
    return scale * (R @ points.T).T + t
def umeyama_alignment(src, dst):
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - mu_src, dst - mu_dst
    cov = dst_centered.T @ src_centered / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    scale = np.trace(np.diag(D)) / ((src_centered ** 2).sum() / src.shape[0])
    t = mu_dst - scale * R @ mu_src
    return scale, R, t
def plot_trajectory_with_gaps(ax, timestamps, traj, color, large_error=None, label=None, linewidth=2.2):
    x, y = traj[:, 0], traj[:, 1]
    isnan = np.isnan(x) | np.isnan(y)
    islarge = large_error if large_error is not None else np.zeros_like(x, dtype=bool)
    segments = []
    current = []
    for i in range(len(x)):
        if not isnan[i] and not islarge[i]:
            current.append((x[i], y[i]))
        else:
            if len(current) > 1:
                segments.append(np.array(current))
            current = []
    if len(current) > 1:
        segments.append(np.array(current))
    for i, seg in enumerate(segments):
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=linewidth, label=label if i == 0 else None)
    # 添加虚线连接缺失段首尾
    i = 0
    while i < len(y):
        if isnan[i]:
            start = i - 1
            while i < len(y) and (isnan[i]):
                i += 1
            end = i
            if 0 <= start < len(y) and end < len(y):
                if (not isnan[start]) and (not isnan[end]) and \
                   (not islarge[start]) and (not islarge[end]):
                    ax.plot([x[start], x[end]], [y[start], y[end]],
                            color=color, linestyle='dashed', linewidth=2.0, alpha=0.8)
        else:
            i += 1
# --- 以上代码保持严格不变 ---
# (新修改) 添加轨迹平滑函数，可以处理包含NaN的轨迹
def smooth_trajectory(traj, window_length=21, polyorder=3):
    """
    使用Savitzky-Golay滤波器平滑轨迹，能处理NaN值。
    - traj: (N, 3) 或 (N, 2) 的轨迹数组
    - window_length: 滤波窗口大小，必须是奇数
    - polyorder: 多项式阶数，必须小于window_length
    """
    smoothed_traj = traj.copy()
    
    # 找到NaN的位置，以便稍后恢复
    nan_mask = np.isnan(traj[:, 0])
    
    # 对XY坐标分别进行处理
    for i in range(2): # 只平滑 X 和 Y
        data = traj[:, i]
        
        # 1. 使用线性插值填充NaN，以便滤波器可以工作
        t = np.arange(len(data))
        valid_mask = ~nan_mask
        if np.sum(valid_mask) < 2: # 如果有效点太少，无法插值，直接返回原轨迹
             return traj
        interp_data = np.interp(t, t[valid_mask], data[valid_mask])
        
        # 2. 应用Savitzky-Golay滤波器
        # mode='interp' 会在边界处产生一些警告，可以忽略或换成 'nearest'
        smoothed_data = savgol_filter(interp_data, window_length, polyorder, mode='nearest')
        
        # 3. 将平滑后的数据放回，并恢复原来的NaN位置
        smoothed_traj[:, i] = smoothed_data
        smoothed_traj[nan_mask, i] = np.nan
        
    return smoothed_traj
# 主流程
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
# == 替换主流程部分（从 seq_list 开始）==
for seq in seq_list:
    # 仅测试一条可固定此行
    seq = 'DJI_20250612194903_0021_V.txt'
    print(f" Processing: {seq}")
    seq_name = seq.split('.')[0]
    
    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))
    for method, color in methods.items():
        file_path = os.path.join(data_root, methods_name[method], seq)
        if not os.path.exists(file_path): continue
        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: continue
        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        valid_mask = ~np.isnan(aligned_xyz[:, 0])
        pos_err = np.linalg.norm(aligned_xyz - poses_gt, axis=1)
        if 'raft' in method:
            for ii in range(len(pos_err)):
                if pos_err[ii] > 10:
                    valid_mask[ii] = False
                    aligned_xyz[ii] = [np.nan,np.nan,np.nan ]
        rel = aligned_xyz.copy()
        rel[valid_mask] -= aligned_xyz[valid_mask][0]
        dx, dy = offset_dict.get(method, (0.0, 0.0))
        rel[valid_mask, 0] += dx
        rel[valid_mask, 1] += dy
        if 'ORB' in method:
            # 这是一个空语句，保持原样
            print(';')
        # if method != "GT":
        #     scale, R, t = umeyama_alignment(rel[timestamps_est[0:300]], poses_gt[timestamps_est[0:300]])
        #     rel = transform_points(rel, scale, R, t)
        
        # (新修改) 对轨迹进行平滑处理
        # window_length和polyorder可以根据需要调整，越大平滑效果越强
        rel_smoothed = smooth_trajectory(rel, window_length=31, polyorder=3)
        # (新修改) 创建带有坐标系的图像
        fig, ax = plt.subplots(figsize=(8, 8)) # 稍微增大图像尺寸以容纳坐标轴
        ax.set_aspect('equal')                 # 保持xy比例
        # (新修改) 添加坐标轴标签和网格
        ax.set_xlabel("North (m)", fontsize=14)
        ax.set_ylabel("East (m)", fontsize=14)
        
        # ax.set_title(f"Trajectory: {methods_name[method]} on {seq_name}", fontsize=16)
        # ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_title(f"Trajectory: {methods_name[method]} on {seq_name}", fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # (新修改) 使用平滑后的轨迹进行绘图
        
        # (新修改) 使用平滑后的轨迹进行绘图 -- 注意：rel_smoothed的0和1索引已交换
        valid = ~np.isnan(rel_smoothed[:, 0])
        ax.plot(rel_smoothed[valid, 1], rel_smoothed[valid, 0], color=color, linewidth=2.5)
        
        # --- 以下部分保持原代码的注释状态 ---
        # ax.plot(rel[:899, 0], rel[:899, 1], color=color, linewidth=2.5)
        
        
        # pos_err[np.isnan(pos_err)] = 0  # 避免 NaN 导致绘图出错
        # plot_trajectory_with_error_circle(ax, rel, pos_err, color=color,
        #                                 linewidth=2.5, alpha=0.25, scale=0.4)
        
        # 保存路径
        save_path = os.path.join(outputs, f"{seq_name}_{methods_name[method]}.png")
        
        # (新修改) 调整保存参数，bbox_inches='tight' 会自动裁剪，适合有坐标轴的图
        # fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        
        # 如果你仍然希望图像是透明背景且无边框，可以恢复原来的保存方式，但坐标轴标签可能会被裁掉
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.0, transparent=True, facecolor='none')
        print(f"✅ Saved: {save_path}")
        plt.close(fig)
print("✅ 每个方法的纯轨迹图已分别保存。")