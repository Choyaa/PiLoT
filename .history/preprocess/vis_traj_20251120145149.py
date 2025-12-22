import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
def vstack_images(img_top, img_bottom, target_width=None):
    """将两个图像在垂直方向拼接，并自动调整宽度一致。"""
    # 自动设定统一宽度为上图宽度
    if target_width is None:
        target_width = img_top.shape[1]

    # 统一 resize 宽度
    def resize_to_width(img, w):
        h = int(img.shape[0] * (w / img.shape[1]))
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    top_resized = resize_to_width(img_top, target_width)
    bottom_resized = resize_to_width(img_bottom, target_width)

    return np.vstack([top_resized, bottom_resized])
def lla_to_enu(lon_deg, lat_deg, alt_m, lon0_deg, lat0_deg, alt0_m):
    R_EARTH = 111320.0
    lat0 = math.radians(lat0_deg)
    east_m  = (lon_deg - lon0_deg) * R_EARTH * math.cos(lat0)
    north_m = (lat_deg - lat0_deg) * R_EARTH
    up_m    = alt_m - alt0_m
    return np.array([east_m, north_m, up_m], float)

def get_fov_corners_enu(apex_enu, yaw_deg, pitch_deg, roll_deg, dist_forward, width, height):
    """
    计算基于Yaw, Pitch, Roll的视锥体角点。
    此版本使用分步旋转，逻辑清晰，确保所有角度按预期工作。
    
    坐标系约定:
    - Yaw:   航向角, 0度=正北, 90度=正东。绕世界Z轴旋转。
    - Pitch: 俯仰角, 正数=抬头。绕相机自身Y轴(右轴)旋转。
    - Roll:  横滚角, 正数=右翼向下。绕相机自身X轴(前轴)旋转。
    """
    
    # 1. 定义一个未经旋转的、标准的相机坐标系
    #    +X_cam = Forward, +Y_cam = Right, +Z_cam = Up
    cam_forward = np.array([1.0, 0.0, 0.0])
    cam_right   = np.array([0.0, 1.0, 0.0])
    cam_up      = np.array([0.0, 0.0, 1.0])
    # 2. 第一步：应用 Yaw (偏航)
    #    航向角0度为北(ENU的+Y)，90度为东(ENU的+X)。这等价于绕世界Z轴(Up)旋转。
    #    我们需要将航向角转换为数学角度 (0度为+X轴，逆时针为正)。
    #    Yaw(0°N) -> Math(90°), Yaw(90°E) -> Math(0°), Yaw(180°S) -> Math(-90°)
    #    转换公式: math_angle = 90 - yaw_deg
    yaw_rot = R.from_euler('z', 90 - yaw_deg, degrees=True)
    
    # 将相机坐标系的三个轴应用Yaw旋转
    current_forward = yaw_rot.apply(cam_forward)
    current_right   = yaw_rot.apply(cam_right)
    current_up      = yaw_rot.apply(cam_up)
    
    # 3. 第二步：应用 Pitch (俯仰)
    #    绕着相机当前的 "right" 轴进行旋转。
    #    注意：pitch_deg为正(抬头)，是绕`right`轴的负向旋转。
    pitch_rot = R.from_rotvec(-pitch_deg * current_right, degrees=True)
    
    # 更新forward和up轴 (right轴作为旋转轴不变)
    current_forward = pitch_rot.apply(current_forward)
    current_up      = pitch_rot.apply(current_up)
    
    # 4. 第三步：应用 Roll (横滚)
    #    绕着相机当前的 "forward" 轴进行旋转。
    #    注意：roll_deg为正(右翼向下)，是绕`forward`轴的正向旋转。
    roll_rot = R.from_rotvec(roll_deg * current_forward, degrees=True)
    
    # 更新right和up轴 (forward轴作为旋转轴不变)
    current_right = roll_rot.apply(current_right)
    current_up    = roll_rot.apply(current_up)
    # 5. 至此，我们得到了最终姿态下，相机的前、右、上向量
    #    使用这些向量构建视锥体
    
    # 视锥体远平面的中心点
    center_far_plane = apex_enu + current_forward * dist_forward
    hw = width / 2.0
    hh = height / 2.0
    
    # 使用最终的 right 和 up 向量来定位角点
    p1 = center_far_plane + hw * current_right + hh * current_up  # Top-Right
    p2 = center_far_plane + hw * current_right - hh * current_up  # Bottom-Right
    p3 = center_far_plane - hw * current_right - hh * current_up  # Bottom-Left
    p4 = center_far_plane - hw * current_right + hh * current_up  # Top-Left
    
    # 返回角点顺序：TR, BR, BL, TL (方便绘制矩形)
    return [p1, p2, p3, p4]

def camera_direction_enu(yaw_deg, pitch_deg, roll_deg=0.0):
    pitch_eff = 90 - pitch_deg
    yaw_eff   = yaw_deg - 90
    rot = R.from_euler('ZYX', [yaw_eff, pitch_eff, roll_deg], degrees=True)
    v = rot.apply(np.array([1,0,0], float))
    return v / np.linalg.norm(v)

class TrajectoryDrawer:
    def __init__(self, lon0, lat0, alt0, 
                 canvas_size=512,
                 fov_dist=20, fov_w=9, fov_h=16,
                 # ---- [新参数] ----
                 view_padding_ratio=0.2, # 默认20%的留白
                 min_view_range_m=350       ,
                 view_center_bias_y=-0.3): # 负值将画面上移
        self.canvas = np.zeros((288, canvas_size, 3), np.uint8)
        self.points = []
        self.canvas_size = canvas_size
        self.lon0, self.lat0, self.alt0 = lon0, lat0, alt0
        self.fov_dist, self.fov_w, self.fov_h = fov_dist, fov_w, fov_h
        
        self.view_padding_ratio = view_padding_ratio
        self.min_view_range_m = min_view_range_m
        # 保存新的偏置参数
        self.view_center_bias_y = view_center_bias_y
    def _normalize_to_canvas(self, x, y, x_min, x_max, y_min, y_max):
        """[修改] 将 ENU 坐标根据动态边界、留白和偏置进行映射"""
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        base_range = max(x_range, y_range, self.min_view_range_m)
        padded_range = base_range * (1 + self.view_padding_ratio)
        if padded_range == 0: padded_range = 1
        # 1. 计算原始的几何中心
        x_center_geo = (x_max + x_min) / 2
        y_center_geo = (y_max + y_min) / 2
        # 2. ---- [核心修改] 根据偏置参数调整中心点 ----
        # 在Y轴上增加一个偏移量，这个偏移量是总视野范围的一部分
        # view_center_bias_y 为负，y_center_eff 会减小，导致世界坐标系下移，画布内容上移。
        y_shift = padded_range * self.view_center_bias_y
        x_center_eff = x_center_geo # X轴通常不需要偏移
        y_center_eff = y_center_geo + y_shift
        # ---------------------------------------------
        # 3. 根据调整后的有效中心(effective center)计算地图边界
        map_min_x = x_center_eff - padded_range / 2
        map_min_y = y_center_eff - padded_range / 2
        # 4. 执行映射
        nx = int((x - map_min_x) / padded_range * self.canvas_size)
        ny = int((y - map_min_y) / padded_range * self.canvas_size)
        
        # Y轴翻转
        ny = self.canvas_size - ny  
        return nx, ny
    # update 方法完全不需要修改，它会自动调用新的 _normalize_to_canvas
    def update(self, lon, lat, alt, roll, pitch, yaw):
        p_enu = lla_to_enu(lon, lat, alt, self.lon0, self.lat0, self.alt0)
        self.points.append((p_enu, roll, pitch, yaw))
        self.canvas[:] = 0
        if not self.points:
            return self.canvas
        all_x = [p[0][0] for p in self.points]
        all_y = [p[0][1] for p in self.points]
        
        p_cur, r, p, y = self.points[-1]
        fov_corners = get_fov_corners_enu(p_cur, y, p, r, self.fov_dist, self.fov_w, self.fov_h)
        for corner in fov_corners:
            all_x.append(corner[0])
            all_y.append(corner[1])
            
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                p0, p1 = self.points[i-1][0], self.points[i][0]
                x0,y0 = self._normalize_to_canvas(p0[0], p0[1], x_min, x_max, y_min, y_max)
                x1,y1 = self._normalize_to_canvas(p1[0], p1[1], x_min, x_max, y_min, y_max)
                cv2.line(self.canvas, (x0,y0), (x1,y1), (255, 220, 100), 2)
        x0,y0 = self._normalize_to_canvas(p_cur[0], p_cur[1], x_min, x_max, y_min, y_max)
        base4_xy = [self._normalize_to_canvas(p[0], p[1], x_min, x_max, y_min, y_max) for p in fov_corners]
        for bx,by in base4_xy:
            cv2.line(self.canvas, (x0,y0), (bx,by), (0,255,255), 1, cv2.LINE_AA) 
        for i in range(4):
            xA,yA = base4_xy[i]
            xB,yB = base4_xy[(i+1)%4]
            cv2.line(self.canvas, (xA,yA), (xB,yB), (0,255,255), 1, cv2.LINE_AA)
        return self.canvas
