import os
import numpy as np
import cv2
from transform import WGS84_to_ECEF, WGS84_to_ECEF_tensor, calculate_matrix_from_precomputed_t, get_matrix
from get_depth import get_points2D_ECEF_projection_v2

# --- 辅助函数部分 (大部分保持不变或微调) ---

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
def draw_legend(image, method_styles, start_pos=(50, 50)):
    """在图像上绘制图例，解释不同形状代表的方法。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = start_pos
    line_height = 40
    # 绘制一个半透明的背景框
    legend_entries = list(method_styles.items())
    max_text_width = 0
    for name, _ in legend_entries:
        (w, h), _ = cv2.getTextSize(name, font, 0.8, 2)
        if w > max_text_width:
            max_text_width = w
            
    bg_width = 80 + max_text_width
    bg_height = len(legend_entries) * line_height + 20
    
    overlay = image.copy()
    cv2.rectangle(overlay, (x - 10, y - 10), (x + bg_width, y + bg_height - 20), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    for name, style in legend_entries:
        shape = style['shape']
        color = style['color']
        
        # 绘制形状样本
        center = (x + 20, y + int(line_height / 2) - 10)
        if shape == 'circle':
            cv2.circle(image, center, 10, color, -1)
        elif shape == 'square':
            cv2.rectangle(image, (center[0] - 10, center[1] - 10), (center[0] + 10, center[1] + 10), color, -1)
        elif shape == 'triangle':
            pts = np.array([[center[0], center[1] - 10], [center[0] - 10, center[1] + 10], [center[0] + 10, center[1] + 10]], np.int32)
            cv2.drawContours(image, [pts], 0, color, -1)
        
        # 绘制方法名称
        cv2.putText(image, name, (x + 50, y + int(line_height / 2) - 5), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        y += line_height
    return image

def hex_to_bgr(hex_color):
    """
    将 #RRGGBB 格式的十六进制颜色字符串转换为 (B, G, R) 元组。
    """
    hex_color = hex_color.lstrip('#') # 去掉开头的 '#'
    # 将 RR, GG, BB 分别转换为整数
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r) # 返回 OpenCV 使用的 BGR 顺序

# (新修改) 重写绘图函数以支持多方法
def draw_points_on_image_multimethod(image, multi_method_xy, multi_method_error, method_styles, point_radius=12):
    """
    在单张图上绘制多个方法在不同目标上的投影点。
    - image: 背景图
    - multi_method_xy: {method_name: {target_name: (x, y)}}
    - multi_method_error: {method_name: {target_name: error}}
    - method_styles: {method_name: {'shape': 'circle', 'color': (B,G,R)}}
    """
    img = to_grayscale_bg(image.copy())
    font = cv2.FONT_HERSHEY_DUPLEX

    # 遍历每个方法进行绘制
    for method_name, xy_dict in multi_method_xy.items():
        if method_name not in multi_method_error or method_name not in method_styles:
            print(f"Skipping {method_name}: missing error data or style info.")
            continue
        
        error_dict = multi_method_error[method_name]
        style = method_styles[method_name]
        shape = style['shape']
        color = style['color'] # 固定颜色

        # 遍历该方法的所有目标点
        target_name = 'car'
        coord = xy_dict
        
        x, y = int(coord[0]), int(coord[1])
        
        # 绘制点（根据方法赋予不同形状）
        if shape == 'circle':
            cv2.circle(img, (x, y), point_radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), point_radius, (0, 0, 0), 2, cv2.LINE_AA) # 加一个黑色描边
        elif shape == 'square':
            cv2.rectangle(img, (x - point_radius, y - point_radius), (x + point_radius, y + point_radius), color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x - point_radius, y - point_radius), (x + point_radius, y + point_radius), (0, 0, 0), 2, cv2.LINE_AA)
        elif shape == 'triangle':
            pts = np.array([
                [x, y - point_radius], 
                [x - point_radius, y + point_radius], 
                [x + point_radius, y + point_radius]
            ], np.int32)
            cv2.drawContours(img, [pts], 0, color, -1, cv2.LINE_AA)
            cv2.drawContours(img, [pts], 0, (0, 0, 0), 2, cv2.LINE_AA)
        elif shape == 'diamond':
            # 菱形有四个顶点：上、右、下、左
            pts = np.array([
                [x, y - point_radius],                  # 顶点 (Top)
                [x + point_radius, y],                  # 右顶点 (Right)
                [x, y + point_radius],                  # 底点 (Bottom)
                [x - point_radius, y]                   # 左顶点 (Left)
            ], np.int32)
            # 绘制实心菱形和描边
            cv2.drawContours(img, [pts], 0, color, -1, cv2.LINE_AA)
            cv2.drawContours(img, [pts], 0, (0, 0, 0), 2, cv2.LINE_AA)
        elif shape == 'star':
            # 绘制五角星需要计算10个顶点（5个外顶点，5个内顶点）
            pts = []
            outer_r = point_radius
            inner_r = int(point_radius * 0.4) # 内圈半径，0.4 是一个比较好看的比例
            
            for i in range(10):
                angle = np.pi / 2 + i * np.pi / 5  # 从90度开始，每次转36度
                r = outer_r if i % 2 == 0 else inner_r # 偶数点用外半径，奇数点用内半径
                
                pt_x = int(x - r * np.cos(angle))
                pt_y = int(y - r * np.sin(angle))
                pts.append([pt_x, pt_y])
                
            pts = np.array(pts, np.int32)
            # 绘制实心星星和描边
            cv2.drawContours(img, [pts], 0, color, -1, cv2.LINE_AA)
            cv2.drawContours(img, [pts], 0, (0, 0, 0), 2, cv2.LINE_AA)
        
        # 在点旁边标注目标ID，方便识别
        cv2.putText(img, target_name, (x + point_radius + 5, y + 5), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 绘制图例h, w, _ = img.shape
    
    # 定义目标裁切尺寸
    crop_w, crop_h = 960, 540
    # 计算裁切区域的左上角坐标 (cx - w/2, cy - h/2)
    h, w, _ = img.shape
    start_x = w // 2 - crop_w // 2
    start_y = h // 2 - crop_h // 2
    
    # 计算裁切区域的右下角坐标
    end_x = start_x + crop_w
    end_y = start_y + crop_h
    
    # 执行裁切 (使用 NumPy 的切片功能)
    # 格式: [y_start:y_end, x_start:x_end]
    cropped_img = img[start_y:end_y, start_x:end_x]
    img = draw_legend(cropped_img, method_styles)

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
         'Render2loc': 'Render2loc',
         "ORB@per30": "Render2ORB"
         
    }
    #     "#007F49",   # GeoPixel - 鲜红
    # "#86AED5",   # PixLoc - 蓝
    # "#EF6C5D",   # Render2Loc - 绿
    # "#C79ACD",   # Render2ORB - 紫
    # "#F7B84A",   # Render2RAFT - 橙
    method_styles = {
    'GT':              {'shape': 'circle',   'color': (0, 255, 0)},  # 绿色 (注意我修正了您注释的颜色)
    'Render2loc@raft': {'shape': 'square',   'color': hex_to_bgr("#F7B84A")}, # (74, 184, 247)
    'FPVLoc':          {'shape': 'star',     'color': hex_to_bgr('#007F49')}, # (73, 127, 0)
    'Pixloc':          {'shape': 'circle',   'color': hex_to_bgr("#86AED5")}, # (213, 174, 134)
    'Render2loc':      {'shape': 'triangle', 'color': hex_to_bgr("#EF6C5D")}, # (93, 108, 239)
    'Render2ORB':      {'shape': 'diamond', 'color': hex_to_bgr("#C79ACD")}  # (205, 154, 199)
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
            multi_method_xyz, # 传递xyz数据用于计算error（如果需要的话，此处暂不需要）
            method_styles
        )

        # 5. 保存结果
        out_path = os.path.join(out_dir, f"{seq}_{VIEW_FRAME_IMG_NAME}_combined_vis.png")
        cv2.imwrite(out_path, vis_img)
        print(f"\n[SUCCESS] Combined visualization saved to: {out_path}")

if __name__ == "__main__":
    main()
