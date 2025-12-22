import os
import numpy as np
import cv2
from transform import WGS84_to_ECEF, WGS84_to_ECEF_tensor, calculate_matrix_from_precomputed_t, get_matrix
from get_depth import get_points2D_ECEF_projection_v2

# --- 辅助函数部分 (大部分保持不变或微调) ---
OUTER_SHAPE = 'square'         # 外框形状: 'square' 或 'circle'
OUTER_THICKNESS = 2            # 外框的线条粗细
INNER_SHAPE = 'circle'         # 内点形状
INNER_COLOR = (0, 0, 0)        # 内点颜色：黑色 (B,G,R 格式)
# --- 2. 定义每个方法独有的颜色 ---
# 字典只负责颜色区分，注意 OpenCV 使用 (B, G, R) 顺序
method_color_map = {
    'GT':              (0, 255, 0),      # 绿色
    'Render2loc@raft': (0, 0, 255),      # 红色
    'FPVLoc':          (255, 255, 0),    # 青色
    'Pixloc':          (255, 0, 255),    # 品红
    'Render2loc':      (255, 0, 0),      # 蓝色
    'Render2ORB':      (0, 165, 255),    # 橙色
}
def to_grayscale_bg(image, descale=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    if descale:
        enhanced = cv2.normalize(enhanced, None, alpha=30, beta=220, norm_type=cv2.NORM_MINMAX)
    bw = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return bw

def normalize_error_dict_fixed_range(error_dict, min_clip=0.0, max_clip=5.0, eps=1e-6):
    norm_dict = {}
    if not error_dict: return norm_dict
    for k, v in error_dict.items():
        if not np.isfinite(v): continue
        clipped = np.clip(v, min_clip, max_clip)
        norm = (clipped - min_clip) / (max_clip - min_clip) if (max_clip - min_clip) > eps else 0.5
        norm_dict[k] = float(norm)
    for k, v in error_dict.items():
        if k not in norm_dict: norm_dict[k] = 0.0
    return norm_dict

# (新修改) 绘制图例的辅助函数
def draw_legend(image, color_map, top_left=(20, 20), icon_size=10, spacing=8):
    """
    在图像上绘制与新样式匹配的图例。
    - image: 待绘制的图像
    - color_map: {method_name: (B, G, R)}
    - top_left: 图例区域的左上角坐标
    - icon_size: 图例中图标的半边长
    - spacing: 图标与文字、行与行之间的间距
    """
    img = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    
    x, y = top_left
    
    for method_name, color in color_map.items():
        # --- 绘制图例图标 ---
        icon_center_x = x + icon_size
        icon_center_y = y + icon_size
        
        # 1. 绘制外框
        if OUTER_SHAPE == 'square':
            cv2.rectangle(
                img, 
                (x, y), 
                (x + icon_size * 2, y + icon_size * 2), 
                color, 
                OUTER_THICKNESS, 
                cv2.LINE_AA
            )
        elif OUTER_SHAPE == 'circle':
            cv2.circle(img, (icon_center_x, icon_center_y), icon_size, color, OUTER_THICKNESS, cv2.LINE_AA)

        # 2. 绘制内点
        inner_radius = max(1, icon_size // 3)
        if INNER_SHAPE == 'circle':
            cv2.circle(img, (icon_center_x, icon_center_y), inner_radius, color, -1, cv2.LINE_AA)

        # --- 绘制文字 ---
        text_x = x + icon_size * 2 + spacing
        text_y = y + icon_size + 5 # 调整y坐标使文字与图标中心对齐
        cv2.putText(img, method_name, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        # 移动到下一行
        y += icon_size * 2 + spacing
        
    return img



# (新修改) 重写绘图函数以支持多方法
def draw_points_on_image_multimethod(image, multi_method_xy, method_color_map, point_radius=10):
    """
    在单张图上绘制多个方法的投影点，所有点都采用“外框+内点”的样式。
    - image: 背景图
    - multi_method_xy: {method_name: (x, y)}
    - method_color_map: {method_name: (B, G, R)}
    - point_radius: 外框的半径或半边长
    """
    # 辅助函数，如果需要可以将背景转为灰度以突出彩色点
    # img = to_grayscale_bg(image.copy()) 
    img = image.copy()
    font = cv2.FONT_HERSHEY_DUPLEX

    # ---- 核心绘图循环 ----
    for method_name, coord in multi_method_xy.items():
        # 从配置中获取颜色，如果找不到则使用默认的灰色
        color = method_color_map.get(method_name)
        if color is None:
            print(f"Skipping {method_name}: missing color info in method_color_map.")
            continue
        
        x, y = int(coord[0]), int(coord[1])
        print(method_name, x, y)
        # --- 1. 绘制外框 (空心) ---
        if OUTER_SHAPE == 'square':
            cv2.rectangle(
                img,                                          # 图像
                (x - point_radius, y - point_radius),         # 左上角
                (x + point_radius, y + point_radius),         # 右下角
                color,                                        # 边框颜色
                thickness=OUTER_THICKNESS,                    # >0 的值表示空心
                lineType=cv2.LINE_AA
            )
        elif OUTER_SHAPE == 'circle':
            cv2.circle(img, (x, y), point_radius, color, OUTER_THICKNESS, cv2.LINE_AA)
        
        # --- 2. 绘制内点 (实心) ---
        # 让内点的大小与外框关联，看起来更协调
        inner_radius = max(1, point_radius // 3) 
        if INNER_SHAPE == 'circle':
            cv2.circle(
                img, (x, y), inner_radius, color, -1, cv2.LINE_AA) # thickness=-1 表示实心
        
        # 在点旁边标注目标ID (您的原始逻辑)
        target_name = 'car' # 您的代码中是固定的
        # cv2.putText(img, target_name, (x + point_radius + 5, y + 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 绘制图例 (使用新的配套函数)
    # 获取原始图像的尺寸
    h, w, _ = img.shape
    # 定义目标裁切尺寸
    crop_w, crop_h = 960, 540
    # 计算裁切区域的左上角坐标 (cx - w/2, cy - h/2)
    start_x = w // 2 - crop_w // 2
    start_y = h // 2 - crop_h // 2
    
    # 计算裁切区域的右下角坐标
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    # 执行裁切 (使用 NumPy 的切片功能)
    # 格式: [y_start:y_end, x_start:x_end]
    cropped_img = img[start_y:end_y, start_x:end_x]
    img = draw_legend(cropped_img, method_color_map)

    return img


def get_poses(pose_file, img_name):
    # 此函数保持原样，仅用于获取特定帧的相机位姿
    with open(pose_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if parts and parts[0] == img_name:
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1:])
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                return get_matrix(translation, euler_angles)
    return None

def get_es_xyz(save_xyz_path):
    # 此函数只读取XYZ数据，返回 {car_name: {img_name: [x,y,z]}}
    xyz_dict = {}
    if not os.path.exists(save_xyz_path):
        return xyz_dict
    import csv
    with open(save_xyz_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        for parts in reader:
            if len(parts) >= 4:
                img_name = parts[0]
                # 这里假设目标点是固定的，只取一个car的所有img_name
                # 为了简化，我们只处理一个car_name的情况
                xyz_dict[img_name] = list(map(float, parts[1:4]))
    return xyz_dict


# (新修改) 重构 main 函数
def main():
    base_path = "/mnt/sda/MapScape/query/estimation/position_result"
    image_path = "/mnt/sda/MapScape/query/images"
    target_pos_path = "/mnt/sda/MapScape/query/estimation/position_result"
    pose_path = "/mnt/sda/MapScape/query/estimation/result_images"
    
    out_dir = "/mnt/sda/MapScape/query/estimation/position_result/error_combined"
    os.makedirs(out_dir, exist_ok=True)
    
    # 定义方法和它们的样式
    methods = {
         'GT': 'GT',
         'Render2loc@raft': 'Render2RAFT',
         'FPVLoc': 'GeoPixel',
         'Pixloc': 'PixLoc',
         'Render2loc': 'Render2loc'
    }
    
    

    rcamera = [3840, 2160, 2700.0, 2700.0, 1915.7, 1075.1]
    
    txt_files = [] # 只处理一个序列
    seq = 'DJI_20250612194903_0021_V'
    # 待可视化的图像帧和目标ID
    VIEW_FRAME_IMG_NAME = '555_0.png'
    TARGET_CAR_NAME = '1' # 假设我们只关心ID为'1'的目标车
    
    for view in range(800):
        print(f"\n-------- Processing Sequence: {seq} --------")
        VIEW_FRAME_IMG_NAME = str(view)+'_0.png'
        # 1. 加载背景图和相机位姿
        full_image_path = os.path.join(image_path, seq, VIEW_FRAME_IMG_NAME)
        image = cv2.imread(full_image_path)
        if image is None:
            print(f"[ERROR] Failed to read image: {full_image_path}")
            continue

        
        
        # if T_c2w is None:
        #     print(f"[ERROR] Failed to get camera pose for {VIEW_FRAME_IMG_NAME} in {seq}")
        #     continue

        # 2. 准备存储所有方法数据的容器
        multi_method_xyz = {}
        multi_method_xy_proj = {}
        
        # 3. 数据先行：循环加载所有方法的数据
        for method_key, subfolder in methods.items():
            print(f"  -> Loading data for: {method_key}")
            
            # 确定结果文件路径
            result_file = os.path.join(target_pos_path, method_key, f"{seq}.txt") 
            
            # 读取XYZ坐标
            xyz_data_all_cars = get_es_xyz(result_file)
            
            # 筛选出我们关心的目标
            target_xyz_wgs = xyz_data_all_cars.get(VIEW_FRAME_IMG_NAME)
            if target_xyz_wgs:
                # 存储WGS84坐标
                multi_method_xyz[method_key] = target_xyz_wgs
                pose_file = os.path.join(pose_path, method_key, f"{seq}.txt")
                T_c2w = get_poses(pose_file, VIEW_FRAME_IMG_NAME)
                # 转换并投影到2D
                xyz_ecef = WGS84_to_ECEF(target_xyz_wgs)
                xy_proj = get_points2D_ECEF_projection_v2(T_c2w, rcamera, np.expand_dims(xyz_ecef, axis=0))
                
                if xy_proj is not None and len(xy_proj) > 0:
                    multi_method_xy_proj[method_key] = xy_proj[0] # 取第一个点
            else:
                print(f"    [WARN] No data for frame '{VIEW_FRAME_IMG_NAME}' for target '{TARGET_CAR_NAME}' in {method_key}")

        # 4. 一次绘图：调用新的绘图函数
        # 注意：这里的error字典是空的，因为我们不再按误差着色，而是按方法区分
        vis_img = draw_points_on_image_multimethod(
            image.copy(), 
            multi_method_xy_proj, 
            # multi_method_xyz,# 传递xyz数据用于计算error（如果需要的话，此处暂不需要）
            method_color_map,  # 传入新的颜色配置
            point_radius=12    # 可以继续使用这个参数控制大小
        )

        # 5. 保存结果
        out_path = os.path.join(out_dir, f"{seq}_{VIEW_FRAME_IMG_NAME}_combined_vis.png")
        cv2.imwrite(out_path, vis_img)
        print(f"\n[SUCCESS] Combined visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
