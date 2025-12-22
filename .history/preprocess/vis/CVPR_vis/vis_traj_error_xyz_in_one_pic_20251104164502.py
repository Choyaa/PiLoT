import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from transform import wgs84tocgcs2000_batch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.signal import savgol_filter

# --- 配置和辅助函数部分 (大部分保持不变) ---
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelweight'] = 'bold'

methods_name = {
    "GT": "Ground Truth",
    "FPVLoc": "PiLoT (Ours)",
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

data_root = "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan"
outputs = os.path.join(data_root, "outputs")
os.makedirs(outputs, exist_ok=True)

# --- 辅助函数 (load_pose*, align_to_full_timestamps, transform_points, smooth_trajectory) 保持不变 ---

def load_pose_with_name(file_path):
    data, angles, timestamps = [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 7:
                name = parts[0]
                if "_" in name: frame_idx = int(name.split("_")[0])
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
    for i, t in enumerate(timestamps_est):
        if t in idx_map:
            aligned[idx_map[t]] = values[i]
    return aligned

def transform_points(points, scale, R, t):
    # 处理NaN值，避免它们参与矩阵运算
    nan_mask = np.isnan(points[:, 0])
    valid_points = points[~nan_mask]
    transformed_points = scale * (R @ valid_points.T).T + t
    
    # 创建一个完整大小的结果数组，并填回NaN
    result = np.full_like(points, np.nan)
    result[~nan_mask] = transformed_points
    return result

def umeyama_alignment(src, dst):
    # 找到两个轨迹中共同有效的点进行对齐
    valid_mask_src = ~np.isnan(src[:, 0])
    valid_mask_dst = ~np.isnan(dst[:, 0])
    common_valid_mask = valid_mask_src & valid_mask_dst
    
    src_common = src[common_valid_mask]
    dst_common = dst[common_valid_mask]

    # 如果共同有效的点少于2个，无法对齐，返回单位变换
    if src_common.shape[0] < 2:
        print("Warning: Not enough common valid points for Umeyama alignment. Returning identity transform.")
        return 1.0, np.eye(3), np.zeros(3)

    mu_src, mu_dst = src_common.mean(0), dst_common.mean(0)
    src_centered, dst_centered = src_common - mu_src, dst_common - mu_dst
    cov = dst_centered.T @ src_centered / src_common.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    # 计算尺度
    var_src = np.var(src_centered, axis=0).sum()
    scale = 1.0 / var_src * np.trace(np.diag(D)) if var_src > 1e-9 else 1.0
    
    t = mu_dst - scale * R @ mu_src
    return scale, R, t

def smooth_trajectory(traj, window_length=21, polyorder=3):
    smoothed_traj = traj.copy()
    nan_mask = np.isnan(traj[:, 0])
    for i in range(2): # 只平滑 X 和 Y
        data = traj[:, i]
        t = np.arange(len(data))
        valid_mask = ~nan_mask
        if np.sum(valid_mask) < polyorder + 1: return traj # 有效点太少无法滤波
        interp_data = np.interp(t, t[valid_mask], data[valid_mask])
        smoothed_data = savgol_filter(interp_data, window_length, polyorder, mode='nearest')
        smoothed_traj[nan_mask, i] = np.nan
        smoothed_traj[~nan_mask, i] = smoothed_data[~nan_mask]
    return smoothed_traj

def plot_trajectory_with_gaps(ax, traj, color, large_error_mask=None, label=None, linewidth=2.2, zorder=1):
    x, y = traj[:, 0], traj[:, 1]
    is_invalid = np.isnan(x)
    if large_error_mask is not None:
        is_invalid = is_invalid | large_error_mask

    segments = []
    current_segment = []
    for i in range(len(x)):
        if not is_invalid[i]:
            current_segment.append((x[i], y[i]))
        else:
            if len(current_segment) > 1: segments.append(np.array(current_segment))
            current_segment = []
    if len(current_segment) > 1: segments.append(np.array(current_segment))

    for i, seg in enumerate(segments):
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=linewidth, label=label if i == 0 else None, zorder=zorder)


# --- (新修改) 重构主流程，用于生成叠加图 ---
seq_list = sorted([f for f in os.listdir(os.path.join(data_root, "GT")) if f.endswith(".txt")])

for seq in seq_list:
    seq = 'DJI_20250612194903_0021_V.txt' # 如果只想测试一个序列，可以固定
    print(f"�� Processing sequence for combined plot: {seq}")
    seq_name = seq.split('.')[0]
    
    # 1. 创建一张图，所有轨迹都将画在这上面
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlabel("East (m)", fontsize=14, weight='bold')
    ax.set_ylabel("North (m)", fontsize=14, weight='bold')
    ax.set_title(f"Geo-localization Trajectories on {seq_name}", fontsize=16, weight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)

    # 2. 首先加载 GT 数据作为基准
    poses_gt, _ = load_pose(os.path.join(data_root, "GT", seq))
    full_timestamps = np.arange(len(poses_gt))
    
    # 加载并存储所有方法的轨迹数据
    all_poses = {}
    for method, color in methods.items():
        # GT数据已经加载
        if method == "GT":
            all_poses[method] = poses_gt
            continue

        file_path = os.path.join(data_root, methods_name[method].split(" ")[0], seq)
        if not os.path.exists(file_path): 
            print(f"⚠️ File not found for {method}, skipping.")
            continue

        timestamps_est, poses_est, _ = load_pose_with_name(file_path)
        if len(poses_est) == 0: 
            print(f"⚠️ No poses found for {method}, skipping.")
            continue
        
        # 将估计的位姿对齐到完整的时间戳上
        all_poses[method] = align_to_full_timestamps(full_timestamps, timestamps_est, poses_est)

    # 3. 循环绘图：对齐、平滑、然后绘制到同一张图上
    for method, color in methods.items():
        if method not in all_poses:
            continue
        
        poses_to_plot = all_poses[method]

        # 对齐：所有方法都向GT对齐
        if method != "GT": 
            scale, R, t = umeyama_alignment(poses_to_plot, poses_gt)
            poses_to_plot = transform_points(poses_to_plot, scale, R, t)

        # 计算位置误差，用于'Render2RAFT'的过滤
        pos_err = np.linalg.norm(poses_to_plot - poses_gt, axis=1)

        # 平滑处理
        poses_smoothed = smooth_trajectory(poses_to_plot, window_length=31, polyorder=3)
        
        # 定义线宽和Z-order，让 'PiLoT' 和 'GT' 更突出
        linewidth = 4.0 if 'FPVLoc' in method else 2.5
        zorder = 10 if 'FPVLoc' in method else (5 if method != 'GT' else 9) # FPVLoc最顶层, GT次之, 其他在下

        # (新) 条件性绘图，为Render2RAFT过滤掉大误差点
        large_error_mask = None
        if 'Render2loc@raft' in method:
            error_threshold = 10.0
            large_error_mask = pos_err > error_threshold
            large_error_mask[np.isnan(pos_err)] = True # 同时过滤掉NaN
        
        # 使用可以断开的绘图函数
        plot_trajectory_with_gaps(
            ax, 
            poses_smoothed, 
            color=color, 
            large_error_mask=large_error_mask,
            label=methods_name.get(method, method),
            linewidth=linewidth,
            zorder=zorder
        )
        print(f"  -> Plotted: {method}")
        
    # 4. 循环结束后，添加图例
    ax.legend(fontsize=12)

    # 5. 保存最终的叠加图
    save_path = os.path.join(outputs, f"{seq_name}_combined_trajectories.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\n✅ Combined plot saved to: {save_path}")
    
    # 6. 关闭图像，准备处理下一个序列
    plt.close(fig)

print("\n�� All processing finished.")
