import os

import numpy as np
import os
import argparse
import csv
import cv2
from transform import WGS84_to_ECEF, WGS84_to_ECEF_tensor, calculate_matrix_from_precomputed_t, get_matrix
from get_depth import get_points2D_ECEF_projection_v2


def to_grayscale_bg(image, descale=True):
    """把原始图转为黑白底图，保留对比，返回3通道 BGR 用于叠加。"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 可选做一点局部对比增强：自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    if descale:
        # 轻微降低亮度/对比避免抢占注意力
        enhanced = cv2.normalize(enhanced, None, alpha=30, beta=220, norm_type=cv2.NORM_MINMAX)
    bw = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bw
def normalize_error_dict_fixed_range(error_dict, min_clip=0.0, max_clip=5.0, eps=1e-6):
    """
    固定把 error 截断在 [min_clip, max_clip] 后归一化到 [0,1]。
    超出范围的都剪切，避免极端值拉扯配色。
    """
    norm_dict = {}
    if not error_dict:
        return norm_dict
    for k, v in error_dict.items():
        if not np.isfinite(v):
            continue
        clipped = np.clip(v, min_clip, max_clip)
        norm = (clipped - min_clip) / (max_clip - min_clip) if (max_clip - min_clip) > eps else 0.5
        norm_dict[k] = float(norm)
    # 对于非 finite 的保底 0（或你想用别的）
    for k, v in error_dict.items():
        if k not in norm_dict:
            norm_dict[k] = 0.0
    return norm_dict
def draw_error_points_on_image(image, multi_xy_dict, error_t_dict,
                               point_radius=10, thickness=-1,
                               colormap=cv2.COLORMAP_TURBO):
    """
    黑白底 + 误差点（大在上）+ 美化 colorbar（固定 clip 0~5m）+ inf 用空心 + 缺失 xy 提示
    全部 OpenCV 调用用位置参数，避免 new style getargs format 错误。
    """
    img = to_grayscale_bg(image)
    font = cv2.FONT_HERSHEY_DUPLEX
    for name, xy_dict in multi_xy_dict.items():
        error_t = error_t_dict[name]
        if not error_t or not isinstance(error_t, dict) or len(error_t) == 0:
            cv2.putText(img, "No valid predictions", (10, 30), font, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
            return img

        # 归一化（固定 0~5m）
        norm_err = normalize_error_dict_fixed_range(error_t, min_clip=0.0, max_clip=5.0)

        # 先画小误差、后画大误差（大误差在顶层），inf 视为最大
        sorted_keys = sorted(error_t.keys(), key=lambda k: error_t[k] if np.isfinite(error_t[k]) else float('inf'))

        missing_in_xy = []
        for key in sorted_keys:
            # if error_t[key] > 5: continue
            if key not in xy_dict:
                missing_in_xy.append(key)
                continue
            coord = xy_dict[key]
            if len(coord) == 0:
                continue
            x, y = coord[0]
            if not np.isfinite(error_t[key]): #int(key.split('_')[0])%5 ==0 or
                cv2.circle(img, (int(x), int(y)), point_radius, (0,0,255), 2, cv2.LINE_AA)
                # cv2.putText(img, "?", (int(x) + point_radius, int(y) - point_radius), font, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
                continue
            # noise = np.random.uniform(0, 0.8) 
            # err_norm = np.clip(norm_err.get(key, 0.0)-0.05 ,  0.05, None)
            # err_norm = norm_err.get(key, 0.0)+noise
            err_norm = norm_err.get(key, 0.0)
            
            cmap_val = int(np.round(err_norm * 255))
            color = cv2.applyColorMap(np.array([[cmap_val]], dtype=np.uint8), colormap)[0, 0].tolist()
            # 外发光
            glow_radius = point_radius + 5
            overlay = img.copy()
            cv2.circle(overlay, (int(x), int(y)), glow_radius, color, -1, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
            cv2.circle(img, (int(x), int(y)), point_radius, color, thickness, cv2.LINE_AA)

        # 缺失 xy 提示
        if missing_in_xy:
            sample = missing_in_xy[:5]
            txt = "Missing XY: " + ",".join(sample)
            cv2.rectangle(img, (5, img.shape[0] - 60), (img.shape[1] // 2, img.shape[0] - 5), (0, 0, 0), -1)
            cv2.putText(img, txt, (10, img.shape[0] - 30), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            if len(missing_in_xy) > 5:
                more = f"+{len(missing_in_xy) - 5} more"
                cv2.putText(img, more, (10, img.shape[0] - 12), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    # ---- colorbar（始终画） ----
    h, w = img.shape[:2]
    bar_h = int(h * 0.6)
    bar_w = 40  # 更粗一些
    
    bar_x = w - bar_w - 500
    bar_y = int((h - bar_h) / 2)

    # 梯度从 5m (上) 到 0m (下)
    gradient = np.linspace(255, 0, bar_h, dtype=np.uint8)[:, None]
    gradient = np.repeat(gradient, bar_w, axis=1)
    colored_bar = cv2.applyColorMap(gradient, colormap)

    # 贴 colorbar 本体
    img[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = colored_bar
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (220, 220, 220), 1)

    # 标题和说明
    cv2.putText(img, "Error", (bar_x, bar_y - 25), font, 0.6, (240, 240, 240), 1)
    cv2.putText(img, "clipped to [0,5]m", (bar_x, bar_y - 8), font, 0.35, (200, 200, 200), 1)

    # 映射函数：value 到 y 位置（5m 在上，0m 在下）
    def val_to_y(v):
        t = (5.0 - v) / 5.0  # 5 -> 0, 0 -> 1
        return int(bar_y + np.clip(t, 0, 1) * bar_h)

    # 主刻度：5.0, 2.5, 0.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 6  # 放大字体
    thickness = 16  # 文字粗一点，去掉描边不要再叠加别的颜色
    major_ticks = [5.0, 2.5, 0.0]

    for v in major_ticks:
        y_pos = val_to_y(v)
        # 主刻度线（长）
        cv2.line(img, (bar_x + bar_w + 2, y_pos), (bar_x + bar_w + 18, y_pos), (220, 220, 220), 2)
        label = f"{v:.1f}m"
        (txt_w, txt_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        label_x = bar_x + bar_w + 20
        label_y = y_pos + txt_h // 2

        # 背景块（留一点 padding）
        pad_x = 4
        pad_y = 2
        top_left = (label_x - pad_x, label_y - txt_h - pad_y)
        bottom_right = (label_x + txt_w + pad_x, label_y + pad_y)
        # cv2.rectangle(img, top_left, bottom_right, (30, 30, 30), -1)

        # 只画一次白色文字（无黑边），开抗锯齿
        cv2.putText(img, label, (label_x, label_y), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)

    # 副刻度：4,3,1 米（短线）
    for v in [4.0, 3.0, 1.0]:
        y_pos = val_to_y(v)
        cv2.line(img, (bar_x + bar_w + 2, y_pos), (bar_x + bar_w + 10, y_pos), (180, 180, 180), 1)
    return img
def evaluate_xyz(results, gt, only_localized=False):
    """
    Compare predicted positions (in WGS84) with GT positions (in WGS84),
    convert to ECEF, compute 2D XY distance error.
    Return error statistics in a dict.
    """
    predictions = {}
    test_names = []
    total_num = 0
    test_num = 0

    if not os.path.exists(results):
        print(f"[Warning] 预测文件不存在: {results}")
        return None
    if not os.path.exists(gt):
        print(f"[Warning] GT文件不存在: {gt}")
        return None
    coords_to_convert = []  # 存放所有待转换的 (lat, lon, alt) 坐标
    metadata_list = []      # 存放每个坐标对应的 (car_name, img_name)
    with open(results, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            try:
                # 解析元信息
                car_name = tokens[1]
                img_name = tokens[0].split('/')[-1]
                
                # 解析并准备坐标数据
                t = np.array(tokens[2:5], dtype=float)
                t[-1] = 0  # 高度设为0
                
                # 将数据添加到列表中
                coords_to_convert.append(t)
                metadata_list.append((car_name, img_name))
                
                test_num += 1 # 计数成功解析的行数
            except Exception as e:
                # 如果某一行解析失败，打印错误并跳过
                print(f"[Error] 预测数据解析失败于行: '{line.strip()}'. 错误: {e}")
                continue
    # === 第二步：进行批量转换 ===
    if coords_to_convert:  # 确保列表不为空
        # 将坐标列表转换为一个大的 NumPy 数组
        batch_input = np.array(coords_to_convert)
        
        # 调用批量转换函数
        batch_results_ecef = WGS84_to_ECEF_tensor(batch_input)
        # === 第三步：分发结果到字典中 ===
        for i, (car_name, img_name) in enumerate(metadata_list):
            # 获取对应的转换结果
            t_ecef = batch_results_ecef[i]
            
            # 将结果存入 predictions 字典
            if car_name not in predictions:
                predictions[car_name] = {}
            predictions[car_name][img_name] = t_ecef.cpu().numpy()
    else:
        print("[Info] 没有有效的数据需要处理。")

    gts = {}
    with open(gt, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            try:
                # 解析元信息
                car_name = tokens[1]
                img_name = tokens[0].split('/')[-1]
                if car_name not in test_names:
                    test_names.append(car_name)
                # 解析并准备坐标数据
                t = np.array(tokens[2:5], dtype=float)
                t[-1] = 0  # 高度设为0
                
                # 将数据添加到列表中
                coords_to_convert.append(t)
                metadata_list.append((car_name, img_name))
                
                test_num += 1 # 计数成功解析的行数
            except Exception as e:
                # 如果某一行解析失败，打印错误并跳过
                print(f"[Error] 预测数据解析失败于行: '{line.strip()}'. 错误: {e}")
                continue
    # === 第二步：进行批量转换 ===
    if coords_to_convert:  # 确保列表不为空
        # 将坐标列表转换为一个大的 NumPy 数组
        batch_input = np.array(coords_to_convert)
        
        # 调用批量转换函数
        batch_results_ecef = WGS84_to_ECEF_tensor(batch_input)
        # === 第三步：分发结果到字典中 ===
        for i, (car_name, img_name) in enumerate(metadata_list):
            # 获取对应的转换结果
            t_ecef = batch_results_ecef[i]
            
            # 将结果存入 predictions 字典
            if car_name not in gts:
                gts[car_name] = {}
            gts[car_name][img_name] = t_ecef.cpu().numpy()
    else:
        print("[Info] 没有有效的数据需要处理。")

    errors_t = {}
    for name in test_names:
        errors_t[name] = {}
        for img_name in gts[name].keys():
            if img_name not in predictions[name] or img_name not in gts[name]:
                if only_localized:
                    continue
                errors_t[name][img_name] = np.inf
            else:
                t_gt = np.array(gts[name][img_name])
                t_pred = np.array(predictions[name][img_name])
                e_t = np.linalg.norm(t_pred - t_gt)
                errors_t[name][img_name] = e_t

    return errors_t

def get_es_xyz(save_xyz_path):
    xyz_dict = {}
    with open(save_xyz_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                x, y, z = map(float, parts[1: ])
                xyz_dict[parts[0]] = [x, y, z] # Add the first element to the name list
    return xyz_dict 
def get_es_xyz(save_xyz_path):
    xyz_dict = {}
    with open(save_xyz_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                x, y, z = map(float, parts[2: ])
                car_name = parts[1]
                img_name = parts[0]
                if int(img_name.split('_')[0])>1219: continue
                if car_name not in xyz_dict.keys():
                    xyz_dict[car_name] = {}
                xyz_dict[car_name][img_name] = [x, y, z] # Add the first element to the name list
                # xyz_dict[parts[0]] = [x, y, z] # Add the first element to the name list
    return xyz_dict 
def get_xy(save_xy_path):
    xy_dict = {}
    with open(save_xy_path, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                x, y = map(float, parts[1: ])
                xy_dict[parts[0]] = [[x, y]] # Add the first element to the name list
    return xy_dict 
def get_poses(pose_file, img_name):
    pose_dict = {}
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                pose_dict[parts[0]] = {} # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                # pitch, roll, yaw, lon, lat, alt,  = map(float, parts[1: ])
                
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                pose_dict[parts[0]]['euler_angles'] = euler_angles
                pose_dict[parts[0]]['translation'] = translation
                
                R_c2w = get_matrix(translation, euler_angles)
                if parts[0] == img_name:
                    return R_c2w 
def get_poses_multi(pose_file, img_name):
    pose_dict = {}
    # 假设 pose_dict, R, get_rotation_enu_in_ecef, WGS84_to_ECEF_batch 已经定义
    # --- 第 1 步：创建列表以收集所有需要处理的数据 ---
    keys_list = []
    translations_list = []
    euler_angles_list = []
    with open(pose_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 7:  # 确保行数据足够 (1 key + 6 floats)
                try:
                    key = parts[0]
                    lon, lat, alt, roll, pitch, yaw = map(float, parts[1:7])
                    
                    # 收集计算所需的数据
                    translation = [lon, lat, alt]
                    euler_angles = [pitch, roll, yaw]
                    
                    keys_list.append(key)
                    translations_list.append(translation)
                    euler_angles_list.append(euler_angles)
                    
                    # 预先填充字典，存储原始解析值
                    pose_dict[key] = {
                        'translation': translation,
                        'euler_angles': euler_angles
                    }
                except ValueError:
                    print(f"警告：跳过格式错误的行 -> {line.strip()}")
                    continue
    # --- 第 2 步：进行批量坐标转换 ---
    if translations_list: # 确保列表不为空
        # 将 translations 列表转换为 NumPy 数组
        translations_np = np.array(translations_list)
        
        # 一次性调用批量转换函数
        t_c2w_batch = WGS84_to_ECEF_tensor(translations_np)
        # --- 第 3 步：遍历收集的数据，完成最终计算 ---
        origin_vector = np.array([0, 0, 0]) # 定义 origin
        
        for i in range(len(keys_list)):
            # 从列表中按顺序取出数据
            key = keys_list[i]
            translation = translations_list[i]
            euler_angles = euler_angles_list[i]
            t_c2w = t_c2w_batch[i] # 获取对应的、已批量计算的结果
            
            # 调用新的辅助函数来计算最终的矩阵
            T_matrix = calculate_matrix_from_precomputed_t(
                translation, euler_angles, t_c2w, origin=origin_vector, mode='c2w'
            )
            
            # 将计算出的矩阵存入字典
            pose_dict[key]['T_c2w'] = T_matrix
    else:
        print("文件中没有找到有效数据。")  
    
def main():
    base_path = "/mnt/sda/MapScape/query/estimation/position_result"
    image_path = "/mnt/sda/MapScape/query/images"
    methods = {
        # 'ORB@per30': 'Render2ORB',
        # 'FPVLoc': 'GeoPixel',
        # 'Render2loc': 'Render2Loc',
        'Render2loc@raft': 'Render2RAFT',
        # 'Pixloc': 'PixLoc',
        # 'GT': 'GT'
    }
    rcamera = [3840, 2160, 2700.0, 2700.0, 1915.7, 1075.1]
    # rcamera = [1920, 1080, 1333.5, 1333.5, 960, 540]
    

    results_list = []
    gt_path = "/mnt/sda/MapScape/query/poses/"
    pose_path = "/mnt/sda/MapScape/query/estimation/result_images"
    txt_files = [
        # 'DJI_20250612174308_0001_V',
        #          'DJI_20250612182017_0001_V',
        #          'DJI_20250612182732_0001_V',
        #          'DJI_20250612183852_0005_V',
                #  'DJI_20250612193930_0012_V',
                #  'DJI_20250612194150_0014_V',
                #  'DJI_20250612194622_0018_V',
                 'DJI_20250612194903_0021_V'
                # "feicuiwan_sim_seq7.txt"
                 ]
    out_dir = "/mnt/sda/MapScape/query/estimation/position_result/error"
    os.makedirs(out_dir, exist_ok=True)
    for txt_file in txt_files:
        seq = txt_file.split('.')[0]
        gt_file = os.path.join(gt_path,f"{seq}.txt")
        # save_xy_path = os.path.join("/mnt/sda/MapScape/query/bbox", seq, seq + '_xy.txt')
        
        # xy_dict = get_xy(save_xy_path)  # 需返回 {key: (x, y)}
        print(f"\n-------- Sequence: {seq} --------")
        img_name = '555_0.png'
        image = cv2.imread(os.path.join(image_path, seq, img_name))
        if image is None:
            print(f"[WARN] 读图失败: {seq}")
            continue
        
        for method_name, subfolder in methods.items():
            pose_file = os.path.join(pose_path, 'GT', f"{seq}.txt")
            result_file = os.path.join(base_path, method_name, f"{seq}.txt")
            if 'GT' in method_name:
                result_file = gt_file
                
            T_c2w = get_poses(pose_file, img_name)
            
            xyz_list = get_es_xyz(result_file)
            gt_xy_list = get_es_xyz(gt_file)
            error_t_dict = evaluate_xyz(result_file, gt_file)  # 必须返回 {key: error_value}
            
            es_xy_dict = {}
            gt_xy_dict= {}
            for name, _ in xyz_list.items():
                es_xyz_list = xyz_list[name]
                gt_xyz_list = gt_xy_list[name]
                for img_name, xyz in es_xyz_list.items():
                    xyz = WGS84_to_ECEF(xyz)
                    xyz = np.expand_dims(xyz, axis = 0)
                    xy = get_points2D_ECEF_projection_v2(T_c2w, rcamera, xyz)
                    if name not in es_xy_dict:
                        es_xy_dict[name] ={}
                    es_xy_dict[name][img_name] = xy /0.5
                for img_name, xyz in gt_xyz_list.items():
                    xyz = WGS84_to_ECEF(xyz)
                    xyz = np.expand_dims(xyz, axis = 0)
                    xy = get_points2D_ECEF_projection_v2(T_c2w, rcamera, xyz)
                    if name not in gt_xy_dict:
                        gt_xy_dict[name] = {}
                    gt_xy_dict[name][img_name] = xy/0.5
            image = cv2.resize(image, (int(image.shape[1] / 0.5), int(image.shape[0] / 0.5)))
            vis_img = draw_error_points_on_image(image, gt_xy_dict, error_t_dict)
            print(f"== target : {name} ==")
        
            out_path = os.path.join(out_dir, f"{method_name}_{seq}_{name}_error_vis.png")
            resized_img = cv2.resize(vis_img, (int(vis_img.shape[1] * 0.25), int(vis_img.shape[0] * 0.25)))
            cv2.imwrite(out_path, resized_img)
            print(f"[Saved] {out_path}")
            print('-')

                
if __name__ == "__main__":
    main()
            


