#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from transform import wgs84tocgcs2000_batch, cgcs2000_to_wgs84_batch  # ([(lon,lat,alt)], epsg) -> Nx3

# ========== 全局风格 ==========
rcParams['font.family'] = 'Times New Roman'
rcParams['axes.labelweight'] = 'bold'

# ========== 可编辑区域 ==========
FILES = [
    "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/GT/DJI_20250612182017_0001_V.txt",  # ← 最后一个：RTK-IMU GT
    "/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/GeoPixel/DJI_20250612182017_0001_V.txt",  # 1612
    # "/mnt/sda/MapScape/query/estimation/result_images/Google/GeoPixel/USA_seq5@8@cloudy@300-100@200.txt", #1602
    # "/mnt/sda/MapScape/query/estimation/result_images/Google/GeoPixel/USA_seq5@8@foggy@intensity2@300-100@200.txt",  #1589
    # "/mnt/sda/MapScape/query/estimation/result_images/Google/GeoPixel/USA_seq5@8@night@300-100@intensity2@200.txt",  #1612
    # "/mnt/sda/MapScape/query/estimation/result_images/Google/GeoPixel/USA_seq5@8@sunset@300-100@200.txt",  #1598
    # "/mnt/sda/MapScape/query/estimation/result_images/Google/GeoPixel/switzerland_seq12@8@rainy@200.txt",  #1605
    
    
]
SAVE_DIR = "/mnt/sda/MapScape/query/estimation/vis/"
SAVE_NAME = "DJI_20250612182017_0001_V.png"
USE_ID_UNION = True
AUTO_TRIM_NAN_HEAD = True   # XY/高度/角度各自减去第一有效值
LABELS: Optional[List[str]] = None
COLORS: Optional[List[str]] = None
OFFSETS: Optional[List[Tuple[float, float]]] = None  # 仅作用于 XY

# 平滑参数
SMOOTH_ENABLE = True
SMOOTH_METHOD = "ema"   # 'ema' 或 'ma'
EMA_ALPHA     = 0.2     # ema 平滑强度（越小越平）
MA_WINDOW     = 11      # ma 窗口（奇数）
# =================================

def parse_frame_id(name: str) -> Optional[int]:
    """从 '123_0.png' 取出 123。"""
    if "_" not in name:
        return None
    head = name.split("_")[0]
    if not head.isdigit():
        m = re.search(r"(\d+)", head)
        if not m:
            return None
        head = m.group(1)
    try:
        return int(head)
    except ValueError:
        return None

def load_pose_with_ids_extended(file_path: str):
    """
    读取一条轨迹，返回：
      ids: (N,)
      xyz: (N,3)  平面坐标（CGCS2000/4547）
      height: (N,) 第 4 列 alt
      pitch:  (N,) 第 6 列
      yaw:    (N,) 第 7 列（原始, 统一到[0,360)后再unwrap）
    """
    ids, lonlatalt, height, pitch, yaw = [], [], [], [], []
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            fid = parse_frame_id(parts[0])
            if fid is None:
                continue
            try:
                lon, lat, alt = map(float, parts[1:4])      # 2/3/4 列
                roll, pit, ya  = map(float, parts[4:7])     # 5/6/7 列
            except Exception:
                continue

            # 临时校正（如有需要）
            # if "poses/DJI_20250926150103_0001_V" in file_path:
            #     alt -= 5
            #     pit -= 0.7
            if 'sunset' or 'night' in file_path:
                fid += 10

            # yaw 统一到 [0,360) 便于观感（后面还会 unwrap）
            if ya < 0:
                ya += 360.0

            ids.append(fid)
            lonlatalt.append((lon, lat, alt))
            height.append(alt)
            pitch.append(pit)
            yaw.append(ya)

    if not ids:
        Z = np.zeros((0,), dtype=float)
        return (np.zeros((0,), dtype=int), np.zeros((0,3), dtype=float),
                Z, Z, Z)

    ids = np.asarray(ids, dtype=int)
    xyz = wgs84tocgcs2000_batch(lonlatalt, 4547)   # (N,3)
    height = np.asarray(height, dtype=float)
    pitch  = np.asarray(pitch, dtype=float)
    yaw    = np.asarray(yaw,   dtype=float)

    order = np.argsort(ids)
    return (ids[order], xyz[order], height[order], pitch[order], yaw[order])

# ======= 平滑工具 =======
def _smooth_segment_ma(x: np.ndarray, win: int) -> np.ndarray:
    """对一段 1D 有效数据做滑动平均（无 NaN）。"""
    if win <= 1 or len(x) < 3:
        return x.copy()
    win = max(1, int(win))
    if win % 2 == 0:
        win += 1
    k = win // 2
    pad = np.pad(x, (k, k), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    y = np.convolve(pad, kernel, mode="valid")
    return y.astype(np.float32)

def _smooth_segment_ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """对一段 1D 有效数据做指数滑动平均（无 NaN）。"""
    if not (0.0 < alpha <= 1.0) or len(x) < 2:
        return x.copy()
    y = np.empty_like(x, dtype=np.float32)
    y[0] = x[0]
    a = float(alpha)
    b = 1.0 - a
    for i in range(1, len(x)):
        y[i] = a * x[i] + b * y[i - 1]
    return y

def smooth_1d_with_nans(arr: np.ndarray, method: str = "ema",
                        win: int = 11, alpha: float = 0.2) -> np.ndarray:
    """对 (M,) 序列按连续非 NaN 段进行平滑，NaN 原样保留。"""
    out = arr.astype(np.float32).copy()
    v = ~np.isnan(arr)
    if not v.any():
        return out
    idx = np.where(v)[0]
    # 找连续段
    starts = [idx[0]]
    for i in range(1, len(idx)):
        if idx[i] != idx[i-1] + 1:
            starts.append(idx[i])
    starts.append(idx[-1] + 1)
    # 遍历段
    for s, e in zip(starts[:-1], starts[1:]):
        seg = arr[s:e].astype(np.float32)
        if method == "ma":
            out[s:e] = _smooth_segment_ma(seg, win)
        else:
            out[s:e] = _smooth_segment_ema(seg, alpha)
    return out

def smooth_xy_with_nans(xy: np.ndarray, method: str = "ema",
                        win: int = 11, alpha: float = 0.2) -> np.ndarray:
    """对 (M,3) 的 XY（前两列）逐列做平滑，Z 不动；NaN 段保持。"""
    if xy.size == 0:
        return xy
    out = xy.copy().astype(np.float32)
    out[:, 0] = smooth_1d_with_nans(out[:, 0], method=method, win=win, alpha=alpha)
    out[:, 1] = smooth_1d_with_nans(out[:, 1], method=method, win=win, alpha=alpha)
    return out

# ======= 对齐工具 =======
def align_by_ids_1d(all_ids: np.ndarray, ids: np.ndarray, val: np.ndarray) -> np.ndarray:
    """1D 对齐，返回 (M,)"""
    out = np.full((len(all_ids),), np.nan, dtype=np.float32)
    idx_map = {t: i for i, t in enumerate(all_ids)}
    for i, t in enumerate(ids):
        j = idx_map.get(t, None)
        if j is not None:
            out[j] = val[i]
    return out

def align_by_ids_2d(all_ids: np.ndarray, ids: np.ndarray, val: np.ndarray) -> np.ndarray:
    """2D 对齐，返回 (M, val.shape[1])"""
    out = np.full((len(all_ids), val.shape[1]), np.nan, dtype=np.float32)
    idx_map = {t: i for i, t in enumerate(all_ids)}
    for i, t in enumerate(ids):
        j = idx_map.get(t, None)
        if j is not None:
            out[j] = val[i]
    return out

def auto_colors(n: int) -> List[str]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]
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
def unwrap_deg(x: np.ndarray) -> np.ndarray:
    """角度解包：degree -> rad unwrap -> degree"""
    rad = np.deg2rad(x)
    rad_unwrap = np.unwrap(rad)
    return np.rad2deg(rad_unwrap)

def main(file_paths: List[str],
         save_dir: str,
         save_name: str = "overlay_4panels.png",
         labels: Optional[List[str]] = None,
         colors: Optional[List[str]] = None,
         offsets: Optional[List[Tuple[float, float]]] = None,
         use_id_union: bool = True,
         auto_trim_head: bool = True):
    assert len(file_paths) > 0, "请在 FILES 中填入至少一个 txt 路径"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 读取全部文件
    tracks: Dict[str, Dict[str, np.ndarray]] = {}
    for fp in file_paths:
        ids, xyz, h, pit, ya = load_pose_with_ids_extended(fp)
        if len(ids) == 0:
            print(f"⚠️ 空轨迹，跳过：{fp}")
            continue
        tracks[fp] = dict(ids=ids, xyz=xyz, h=h, pit=pit, ya=ya)
    if not tracks:
        print("❌ 没有可用轨迹，退出")
        return

    keys = list(tracks.keys())

    # 统一时间轴（ID）
    if use_id_union:
        all_ids = sorted(set(np.concatenate([tracks[k]["ids"] for k in keys])))
        all_ids = np.asarray(all_ids, dtype=int)
    else:
        all_ids = tracks[keys[0]]["ids"]
    M = len(all_ids)

    # 对齐为并行数组
    XYs, Hs, PITs, YAWs = [], [], [], []
    first_points = tracks["/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/GT/"+SAVE_NAME.split(".")[0]+".txt"]['xyz'][0]
    for k in keys:
        d = tracks[k]
        xy = align_by_ids_2d(all_ids, d["ids"], d["xyz"])  # (M,3)
        h  = align_by_ids_1d(all_ids, d["ids"], d["h"])    # (M,)
        pt = align_by_ids_1d(all_ids, d["ids"], d["pit"])  # (M,)
        yw = align_by_ids_1d(all_ids, d["ids"], d["ya"])   # (M,)

        # unwrap 角度（仅对有效值做）
        mask_yw = ~np.isnan(yw)
        if mask_yw.any():
            yw[mask_yw] = unwrap_deg(yw[mask_yw])
        mask_pt = ~np.isnan(pt)
        if mask_pt.any():
            pt[mask_pt] = unwrap_deg(pt[mask_pt])

        # 可选：各自减去首个有效值（对比相对变化更直观）
        if auto_trim_head:
            vxy = ~np.isnan(xy[:, 0])
            if vxy.any():
                xy[vxy, :2] -= xy[vxy, :2][0]    # XY 减首点
            
            # 保留绝对高度和角度（如需相对量，可解开下列注释）
            # vh = ~np.isnan(h);  h[vh]  -= h[vh][0]
            # vp = ~np.isnan(pt); pt[vp] -= pt[vp][0]
            # vy = ~np.isnan(yw); yw[vy] -= yw[vy][0]

        XYs.append(xy)
        Hs.append(h)
        PITs.append(pt)
        YAWs.append(yw)

    # ======= 统一平滑（一次性）=======
    if SMOOTH_ENABLE:
        for i in range(len(XYs)):
            XYs[i]  = smooth_xy_with_nans(XYs[i], method=SMOOTH_METHOD,
                                          win=MA_WINDOW, alpha=EMA_ALPHA)
            Hs[i]   = smooth_1d_with_nans(Hs[i],  method=SMOOTH_METHOD,
                                          win=MA_WINDOW, alpha=EMA_ALPHA)
            PITs[i] = smooth_1d_with_nans(PITs[i], method=SMOOTH_METHOD,
                                          win=MA_WINDOW, alpha=EMA_ALPHA)
            YAWs[i] = smooth_1d_with_nans(YAWs[i], method=SMOOTH_METHOD,
                                          win=MA_WINDOW, alpha=EMA_ALPHA)
    gt = np.array(XYs[-1])
    es = np.array(XYs[-2])
    scale, R, t = umeyama_alignment(gt, es)
    gt = transform_points(gt, scale, R, t)
    with open("/mnt/sda/MapScape/query/estimation/result_images/feicuiwan/GT/"+SAVE_NAME.split(".")[0]+".txt", 'w', encoding='utf-8') as f:
        first_points[-1] = 0
        gt_wgs84 = cgcs2000_to_wgs84_batch(gt+first_points, 4547)
        for i in range(len(gt)):
            euler_enu = [0, PITs[-1][i], YAWs[-1][i]+1]
        
            wgs84_coord = gt_wgs84[i].tolist()
            line_data = wgs84_coord + euler_enu
            name_str = str(i) + '_0.png'
            out_str = ' '.join(map(str, line_data))
            f.write(f'{name_str} {out_str}\n')
        
    # print(f"轨迹文件已写入:")
    
    XYs[-1] = gt
    Hs[-1] = gt[:, -1]
    # 绘制参数
    n = len(keys)
    gt_idx = 0  # 约定最后一个是 RTK-IMU GT
    if labels is None:
        labels = [Path(k).stem for k in keys]
    # 标注 GT
    labels[gt_idx] = labels[gt_idx] 

    if colors is None:
        colors = auto_colors(n)
        # GT 统一为黑色
        colors[gt_idx] = "k"
    if offsets is None:
        offsets = [(0.0, 0.0) for _ in keys]

    # 线型与粗细：GT 更醒目
    line_styles = ["-"] * n
    line_styles[gt_idx] = "--"
    line_widths = [2.0] * n
    line_widths[gt_idx] = 3.0

    # === 四宫格绘图 ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 7), constrained_layout=True)
    ax_xy, ax_h = axes[0, 0], axes[0, 1]
    ax_p,  ax_y = axes[1, 0], axes[1, 1]

    # ── 线型与信息分层设置：最后一个是 RTK-IMU GT ──
    n = len(keys)
    gt_idx = 0  # 约定最后一个为 GT
    if labels is None:
        labels = [Path(k).stem for k in keys]
    labels[gt_idx] = labels[gt_idx] + " (GT: RTK-IMU)"

    if colors is None:
        colors = auto_colors(n)
        colors[gt_idx] = "k"  # GT 黑色
    if offsets is None:
        offsets = [(0.0, 0.0) for _ in keys]

    line_styles = ["-"] * n
    line_styles[gt_idx] = "--"         # GT 虚线
    line_widths = [2.0] * n
    line_widths[gt_idx] = 3.0          # GT 更粗
    alphas = [0.92] * n
    alphas[gt_idx] = 1.0               # GT 不透明

    # ── 先收集 XY 有效点用于统一范围（方窗）──
    xy_all = []
    for (xy, offs) in zip(XYs, offsets):
        v = ~np.isnan(xy[:, 0])
        if v.any():
            xx = xy[v, 0] + offs[0]
            yy = xy[v, 1] + offs[1]
            xy_all.append((xx, yy))

    # 计算正方形边界，避免“Y 很扁”
    if len(xy_all) > 0:
        all_x = np.concatenate([a[0] for a in xy_all])
        all_y = np.concatenate([a[1] for a in xy_all])
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        cx = 0.5 * (x_min + x_max)
        cy = 0.5 * (y_min + y_max)
        dx = x_max - x_min
        dy = y_max - y_min
        # 取较大者做半边长，保持正方形可视范围
        half = 0.5 * max(dx, dy)
        pad_ratio = 0.06
        half *= (1.0 + pad_ratio)
        xlim = (cx - half, cx + half)
        ylim = (cy - half, cy + half)
    else:
        xlim = None
        ylim = None

    # ── XY 轨迹（正方形视窗 + 起终点标记）──
    ax_xy.set_xlabel("X (m, CGCS2000, relative)")
    ax_xy.set_ylabel("Y (m, CGCS2000, relative)")
    ax_xy.grid(True, linestyle="--", alpha=0.3)
    ax_xy.set_title("Planar Trajectory (XY)", fontsize=16)

    # 关键：保证绘图区为正方形，视觉与其它子图同尺寸
    # try:
    #     ax_xy.set_box_aspect(1)  # 需要 Matplotlib >= 3.3
    # except Exception:
    #     pass
    # ax_xy.set_aspect("equal", adjustable="box")  # 数据等比例
    dash_patterns = [
    (0, (6, 0)),   # 短线短空
    (0, (6, 0)),   # 稍长线
    (9, (3, 15)),   # 短线长空
    (12, (3, 15)),   # 长线
    (15, (3, 15)),  # 更长线
    (18, (3, 15)),  # 点划线
]

    # for i, (xy, label, color, offset, lw, ls, a_) in enumerate(
    #     zip(XYs, labels, colors, offsets, line_widths, line_styles, alphas)
    # ):
    #     v = ~np.isnan(xy[:, 0])
    #     if not v.any():
    #         continue
    #     xx = xy[v, 0] + offset[0]
    #     yy = xy[v, 1] + offset[1]

    #     # 轨迹线
    #     ax_xy.plot(xx, yy, color=color, linewidth=lw, linestyle=ls, alpha=a_, label=label, rasterized=True)
    
    for i, (xy, label, color, offset, lw, a_) in enumerate(zip(XYs, labels, colors, offsets, line_widths, alphas)):
        v = ~np.isnan(xy[:, 0])
        if not v.any():
            continue
        
        xx = xy[v, 0] + offset[0] 
        yy = xy[v, 1] + offset[1]
        
        
        # label_short = "RTK-IMU GT, sunny, foggy, night, sunset, rainy"
        # label_short = "RTK-IMU GT,sunny, cloudy, foggy, night, sunset"

        parts = label.split('@')
        if len(parts) >= 3:
            label_short = parts[2]  # 第3个字段
        else:
            label_short = label

        # 根据 i 循环选择虚线样式
        dash_style = dash_patterns[i % len(dash_patterns)]

        ax_xy.plot(
            xx, yy,
            color=color,
            linewidth=lw,
            linestyle=dash_style,  # 使用自定义虚线
            alpha=a_,
            rasterized=True
        )
        
        # 创建一个不可见的点，只用于显示图例
        ax_xy.plot([], [], color=color, label=label_short)



        # 统一 XY 的正方形边界

        # 起终点标记（所有曲线都标，小号；GT 更醒目）
        mk_size = 18 if i == gt_idx else 14
        edge_w  = 1.4 if i == gt_idx else 1.0
        # Start：空心圆；End：实心圆
        ax_xy.scatter(xx[0],  yy[0],  s=mk_size, facecolor='none', edgecolor=color, linewidth=edge_w, zorder=5)
        ax_xy.scatter(xx[-1], yy[-1], s=mk_size, facecolor=color,      edgecolor='none',             zorder=5)

    # 统一设置 XY 的正方形边界
    if xlim is not None and ylim is not None:
        ax_xy.set_xlim(*xlim)
        ax_xy.set_ylim(*ylim)
        

    ax_xy.legend(loc="best", frameon=False, fontsize=9)

    # ── 高度 ──
    ax_h.set_xlabel("Frame ID")
    ax_h.set_ylabel("Height (m)")
    ax_h.grid(True, linestyle="--", alpha=0.3)
    ax_h.set_title("Altitude vs. Frame", fontsize=16)
    # ax_h.set_ylim(200, 250)
    # for i, (h, color, lw, ls, a_) in enumerate(zip(Hs, colors, line_widths, line_styles, alphas)):
    #     v = ~np.isnan(h)
    #     if not v.any():
    #         continue
    #     ax_h.plot(all_ids[v], h[v], color=color, linewidth=lw, linestyle=ls, alpha=a_, rasterized=True)
        
    for i, (h, color, lw, a_) in enumerate(zip(Hs, colors, line_widths, alphas)):
        v = ~np.isnan(h)
        if not v.any():
            continue

        # 根据 i 循环选择虚线样式
        dash_style = dash_patterns[i % len(dash_patterns)]

        ax_h.plot(
            all_ids[v], h[v],
            color=color,
            linewidth=lw,
            linestyle=dash_style,  # 使用自定义虚线
            alpha=a_,
            rasterized=True
        )

    # ── Pitch ──
    ax_p.set_xlabel("Frame ID")
    ax_p.set_ylabel("Pitch (deg)")
    ax_p.grid(True, linestyle="--", alpha=0.3)
    ax_p.set_title("Pitch vs. Frame", fontsize=16)
    # ax_p.set_ylim(30, 60)
    # for i, (p, color, lw, ls, a_) in enumerate(zip(PITs, colors, line_widths, line_styles, alphas)):
    #     v = ~np.isnan(p)
    #     if not v.any():
    #         continue
    #     ax_p.plot(all_ids[v], p[v], color=color, linewidth=lw, linestyle=ls, alpha=a_, rasterized=True)
    for i, (p, color, lw, a_) in enumerate(zip(PITs, colors, line_widths, alphas)):
        v = ~np.isnan(p)
        if not v.any():
            continue

        # 根据 i 循环选择虚线样式
        dash_style = dash_patterns[i % len(dash_patterns)]

        ax_p.plot(
            all_ids[v], p[v],
            color=color,
            linewidth=lw,
            linestyle=dash_style,  # 使用自定义虚线
            alpha=a_,
            rasterized=True
        )
    # ── Yaw ──
    ax_y.set_xlabel("Frame ID")
    ax_y.set_ylabel("Yaw (deg)")
    ax_y.grid(True, linestyle="--", alpha=0.3)
    ax_y.set_title("Yaw vs. Frame", fontsize = 16)
    # for i, (y, color, lw, ls, a_) in enumerate(zip(YAWs, colors, line_widths, line_styles, alphas)):
    #     v = ~np.isnan(y)
    #     if not v.any():
    #         continue
    #     ax_y.plot(all_ids[v], y[v], color=color, linewidth=lw, linestyle=ls, alpha=a_, rasterized=True)
    for i, (y, color, lw, a_) in enumerate(zip(YAWs, colors, line_widths, alphas)):
        v = ~np.isnan(y)
        if not v.any():
            continue

        # 根据 i 循环选择虚线样式
        dash_style = dash_patterns[i % len(dash_patterns)]

        ax_y.plot(
            all_ids[v], y[v],
            color=color,
            linewidth=lw,
            linestyle=dash_style,  # 使用自定义虚线
            alpha=a_,
            rasterized=True
        )
    # 保存
    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, SAVE_NAME)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"✅ 已保存：{out_path}")


if __name__ == "__main__":
    main(
        file_paths=FILES,
        save_dir=SAVE_DIR,
        save_name=SAVE_NAME,
        labels=LABELS,
        colors=COLORS,
        offsets=OFFSETS,
        use_id_union=USE_ID_UNION,
        auto_trim_head=AUTO_TRIM_NAN_HEAD,
    )

