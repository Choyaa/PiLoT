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
                 fov_dist=40, fov_w=18, fov_h=32):
        # 注意，画布的高度我从 288 改成了 canvas_size，让它变成正方形
        # 如果你需要长方形，可以再调整，但正方形更容易处理坐标
        self.canvas = np.zeros((canvas_size, canvas_size, 3), np.uint8) 
        self.points = []
        self.canvas_size = canvas_size
        # ENU 原点
        self.lon0 = lon0
        self.lat0 = lat0
        self.alt0 = alt0
        # FOV 形状
        self.fov_dist = fov_dist
        self.fov_w = fov_w
        self.fov_h = fov_h
    def _normalize_to_canvas(self, x, y, x_min, x_max, y_min, y_max):
        """
        [新] 把 ENU 坐标根据动态边界映射到画布坐标。
        为了防止除以零，并且给轨迹留出一些边界，我们增加一个 padding。
        """
        # 计算动态范围，并增加 10% 的边距 padding
        x_range = (x_max - x_min) * 1.1
        y_range = (y_max - y_min) * 1.1
        
        # 避免范围为0时除以0
        if x_range == 0: x_range = 1 
        if y_range == 0: y_range = 1
        # 计算新的中心点
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        # 确定一个统一的、更大的范围，保持长宽比
        max_range = max(x_range, y_range)
        
        # 根据最大范围重新计算边界
        nx_min = x_center - max_range / 2
        ny_min = y_center - max_range / 2
        nx = int((x - nx_min) / max_range * self.canvas_size)
        ny = int((y - ny_min) / max_range * self.canvas_size)
        
        # y 轴翻转，使之符合图像坐标系 (左上角为0,0)
        ny = self.canvas_size - ny  
        return nx, ny
    def update(self, lon, lat, alt, roll, pitch, yaw):
        """[修改] 加入一帧新姿态，并重新绘制轨迹画面"""
        p_enu = lla_to_enu(lon, lat, alt, self.lon0, self.lat0, self.alt0)
        self.points.append((p_enu, roll, pitch, yaw))
        # 清空画布
        self.canvas[:] = 0
        
        if not self.points:
            return self.canvas
        # ---- [核心修改] 计算所有点的动态边界 ----
        all_x = [p[0][0] for p in self.points]
        all_y = [p[0][1] for p in self.points]
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        # ------------------------------------
        # ---- 画轨迹 ----
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                p0 = self.points[i-1][0]
                p1 = self.points[i][0]
                # 使用新的动态边界进行归一化
                x0,y0 = self._normalize_to_canvas(p0[0], p0[1], x_min, x_max, y_min, y_max)
                x1,y1 = self._normalize_to_canvas(p1[0], p1[1], x_min, x_max, y_min, y_max)
                cv2.line(self.canvas, (x0,y0), (x1,y1), (255, 220, 100), 2)
        # ---- 当前位姿 ----
        p_cur, roll, pitch, yaw = self.points[-1]
        x0,y0 = self._normalize_to_canvas(p_cur[0], p_cur[1], x_min, x_max, y_min, y_max)
        
        # ---- FOV 四棱锥 ----
        base4 = get_fov_corners_enu( p_cur, yaw, pitch, roll, self.fov_dist, self.fov_w, self.fov_h)
        base4_xy = [self._normalize_to_canvas(p[0], p[1], x_min, x_max, y_min, y_max) for p in base4]
        # 画四棱锥
        for bx,by in base4_xy:
            cv2.line(self.canvas, (x0,y0), (bx,by), (0,255,255), 1, cv2.LINE_AA) 
        for i in range(4):
            xA,yA = base4_xy[i]
            xB,yB = base4_xy[(i+1)%4]
            cv2.line(self.canvas, (xA,yA), (xB,yB), (0,255,255), 1, cv2.LINE_AA)
        return self.canvas
