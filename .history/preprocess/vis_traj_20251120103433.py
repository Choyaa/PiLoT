import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R
def lla_to_enu(lon_deg, lat_deg, alt_m, lon0_deg, lat0_deg, alt0_m):
    R_EARTH = 111320.0
    lat0 = math.radians(lat0_deg)
    east_m  = (lon_deg - lon0_deg) * R_EARTH * math.cos(lat0)
    north_m = (lat_deg - lat0_deg) * R_EARTH
    up_m    = alt_m - alt0_m
    return np.array([east_m, north_m, up_m], float)


def camera_direction_enu(yaw_deg, pitch_deg, roll_deg=0.0):
    pitch_eff = 90 - pitch_deg
    yaw_eff   = yaw_deg - 90
    rot = R.from_euler('ZYX', [yaw_eff, pitch_eff, roll_deg], degrees=True)
    v = rot.apply(np.array([1,0,0], float))
    return v / np.linalg.norm(v)


class TrajectoryDrawer:
    def __init__(self, lon0, lat0, alt0, 
                 canvas_size=512,
                 fov_dist=20, fov_w=10, fov_h=5):

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


    def _normalize_to_canvas(self, x, y, min_v=-200, max_v=200):
        """把 ENU 坐标映射到画布坐标（简单归一化）"""
        nx = int((x - min_v) / (max_v - min_v) * self.canvas_size)
        ny = int((y - min_v) / (max_v - min_v) * self.canvas_size)
        ny = self.canvas_size - ny  
        return nx, ny


    def update(self, lon, lat, alt, roll, pitch, yaw):
        """加入一帧新姿态，并重新绘制轨迹画面"""
        p_enu = lla_to_enu(lon, lat, alt, self.lon0, self.lat0, self.alt0)
        self.points.append((p_enu, roll, pitch, yaw))

        # 清空画布
        self.canvas[:] = 0

        # ---- 画轨迹线 ----
        if len(self.points) > 1:
            for i in range(1, len(self.points)):
                p0 = self.points[i-1][0]
                p1 = self.points[i][0]
                x0,y0 = self._normalize_to_canvas(p0[0], p0[1])
                x1,y1 = self._normalize_to_canvas(p1[0], p1[1])
                cv2.line(self.canvas, (x0,y0), (x1,y1), (0,255,255), 2)

        # ---- 画当前机体方向（FOV） ----
        p_cur, roll, pitch, yaw = self.points[-1]
        x0,y0 = self._normalize_to_canvas(p_cur[0], p_cur[1])

        dir_enu = camera_direction_enu(yaw, pitch, roll)
        p_far = p_cur + dir_enu * self.fov_dist
        xf, yf = self._normalize_to_canvas(p_far[0], p_far[1])

        cv2.circle(self.canvas, (x0,y0), 5, (0,0,255), -1)
        cv2.line(self.canvas, (x0,y0), (xf,yf), (0,0,255), 2)

        return self.canvas