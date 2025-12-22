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

def get_fov_corners_enu(apex_enu, yaw, pitch, roll, dist_forward, width, height):
    """
    返回 FOV 四棱锥底面的四个 ENU 坐标点。
    """
    dir_enu = camera_direction_enu(yaw, pitch, roll)

    # 底面中心
    center = apex_enu + dir_enu * dist_forward

    # 找到与朝向垂直的两个方向
    tmp = np.array([0,0,1], float)
    if abs(np.dot(dir_enu, tmp)) > 0.99:
        tmp = np.array([1,0,0], float)

    u = np.cross(dir_enu, tmp); u /= np.linalg.norm(u)
    v = np.cross(dir_enu, u);   v /= np.linalg.norm(v)

    hw = width  / 2.0
    hh = height / 2.0

    # 4 个角
    p1 = center + hw*u + hh*v
    p2 = center + hw*u - hh*v
    p3 = center - hw*u - hh*v
    p4 = center - hw*u + hh*v
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
                 fov_dist=40, fov_w=18, fov_h=32,
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
