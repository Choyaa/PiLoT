import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from transform import wgs84tocgcs2000_batch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import savgol_filter

# --- 以下代码保持严格不变 ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelweight'] = 'bold'
methods_name = {
    "GT": "GT", # (新修改) 更新图例名称以增加可读性
    "FPVLoc": "PiLoT (Ours)",   # (新修改) 更新图例名称
    "Pixloc": "PixLoc",
    "Render2loc": "Render2Loc",
    "ORB@per30": "Render2ORB",
    "Render2loc@raft": "Render2RAFT"
}
methods = {
    "GT": "black",
    "FPVLoc": "#007F49",
    "Pixloc": "#86AED5",
    "Render2loc": "#EF6C5D",
    "ORB@per30": "#C79ACD",
    "Render2loc@raft": "#F7B84A"
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
    x, y = traj[:, 0], traj[:, 1]
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(pos_err[i]) or pos_err[i] <= 0:
            continue
        radius = pos_err[i] * scale
        circle = Circle((x[i], y[i]), radius=radius, color=color, alpha=alpha, linewidth=0)
        ax.add_patch(circle)
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
                else: continue
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
    timestamps_est = timestamps_est[600:650]
    for i, t in enumerate(timestamps_est):
        if t in idx_map:
            aligned[idx_map[t]] = values[i]
    return aligned

def transform_points(points, scale, R, t):
    return scale * (R @ points.T).T + t

def umeyama_alignment(src, dst):
    src_valid = src[~np.isnan(src[:, 0])]
    dst_valid = dst[~np.isnan(dst[:, 0])]
    
    # (新修改) 确保对齐的点数一致，取公共部分
    min_len = min(len(src_valid), len(dst_valid))
    src = src_valid[:min_len]
    dst = dst_valid[:min_len]

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


# (新修改) 保持平滑函数不变
def smooth_trajectory(traj, window_length=21, polyorder=3):
    smoothed_traj = traj.copy()
    nan_mask = np.isnan(traj[:, 0])
    for i in range(2):
        data = traj[:, i]
        t = np.arange(len(data))
        valid_mask = ~nan_mask
        if np.sum(valid_mask) < 2: return traj
        interp_data = np.interp(t, t[valid_mask], data[valid_mask])
        smoothed_data = savgol_filter(interp_data, window_length, polyorder, mode='nearest')
        smoothed_traj[:, i] = smoothed_data
        smoothed_traj[nan_mask, i] = np.nan
    return smoothed_traj

# --- 主流程修改 ---

seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])
for seq in seq_list:
    seq = 'DJI_20250612194903_0021_V.txt'
    print(f"�� Processing sequence for combined plot: {seq}")
    seq_name = seq.split('.')[0]
    
    # (新修改) 先加载GT，并以此为基准
    poses_gt, angles_gt = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))
    
    # (新修改) 在所有方法循环之前创建图纸
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlabel("East (m)", fontsize=14, weight='bold')
    ax.set_ylabel("North (m)", fontsize=14, weight='bold')
    ax.set_title(f"Geo-localization Trajectories on {seq_name}", fontsize=16, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # (新修改) 准备一个字典来存储所有方法的pose，以便对齐
    all_poses = {}

    # (新修改) 第一次循环：加载所有数据并进行初步对齐
    for method, color in methods.items():
        file_path = os.path.join(data_root, methods_name.get(method, method), seq) # 使用 .get 更安全
        if not os.path.exists(file_path): 
            print(f"⚠️ File not found for {method}, skipping.")
            continue

        if method == "GT":
            # all_poses[method] = poses_gt
            all_poses[method] = poses_gt[600:650]
            continue

        timestamps_est, poses_est, angles_est = load_pose_with_name(file_path)
        if len(poses_est) == 0: 
            print(f"⚠️ No poses found for {method}, skipping.")
            continue

        # 对齐到GT的时间戳
        aligned_xyz = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)
        all_poses[method] = aligned_xyz

    # (新修改) 第二次循环：对齐所有轨迹到GT并绘图
    for method, color in methods.items():
        if method not in all_poses:
            continue
        
        poses_to_plot = all_poses[method]
        
        # (新修改) 所有轨迹都与GT对齐 (umeyama)
        # if method != "GT" and 'FPVLoc' not in method: # 假设GT和FPVLoc是绝对坐标，无需对齐
        #     # 找到公共的有效帧来进行对齐
        #     valid_mask = ~np.isnan(poses_to_plot[:, 0]) & ~np.isnan(poses_gt[:, 0])
        #     if np.sum(valid_mask) > 10: # 确保有足够的点来对齐
        #         scale, R, t = umeyama_alignment(poses_to_plot[valid_mask], poses_gt[valid_mask])
        #         poses_to_plot = transform_points(poses_to_plot, scale, R, t)
        #     else:
        #         print(f"⚠️ Not enough valid points to align {method}, plotting raw.")

        # 平滑处理
        poses_smoothed = smooth_trajectory(poses_to_plot, window_length=31, polyorder=3)
        
        # 绘图
        valid = ~np.isnan(poses_smoothed[:, 0])
        linewidth = 4.0 if 'FPVLoc' in method else 2.5 # 让自己的方法更突出
        zorder = 10 if 'FPVLoc' in method else 5       # 让自己的方法在最上层
        
        ax.plot(
            poses_smoothed[valid, 0], 
            poses_smoothed[valid, 1], 
            color=color, 
            linewidth=linewidth,
            label=methods_name.get(method, method), # 使用图例名称
            zorder=zorder
        )
        print(f"  -> Plotted: {method}")

    # (新修改) 添加图例
    ax.legend(fontsize=12)

    # (新修改) 保存最终的合成图像
    save_path = os.path.join(outputs, f"{seq_name}_combined_trajectories.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\n✅ Combined plot saved to: {save_path}")
    plt.close(fig) # 关闭图像，防止在循环中重复显示

print("�� All processing finished.")
