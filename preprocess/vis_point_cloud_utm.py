import os
import time
import cv2
import pycolmap
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm
from transform import get_rotation_enu_in_ecef, WGS84_to_ECEF
from transform import ECEF_to_WGS84,qvec2rotmat
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d.art3d import Line3D
import copy
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class QueryLocalizer:
    def __init__(self, config=None):
        self.config = config
        # 初始化其他需要的成员变量
        self.dataset = config['render2loc']['datasets']
        self.outputs = config['render2loc']['results']
        self.save_loc_path = Path(self.dataset) / self.outputs / (f"{iter}_estimated_pose.txt")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def interpolate_depth(self, pos, depth):
        ids = torch.arange(0, pos.shape[0])
        if depth.ndim != 2:
            if depth.ndim == 3:
                depth = depth[:,:,0]
            else:
                raise Exception("Invalid depth image!")
        h, w = depth.size()
        
        i = pos[:, 0]
        j = pos[:, 1]

        # Valid corners, check whether it is out of range
        i_top_left = torch.floor(i).long()
        j_top_left = torch.floor(j).long()
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

        i_top_right = torch.floor(i).long()
        # j_top_right = torch.ceil(j).long()
        j_top_right = torch.floor(j).long()
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

        # i_bottom_left = torch.ceil(i).long()
        i_bottom_left = torch.floor(i).long()
        
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

        # i_bottom_right = torch.ceil(i).long()
        # j_bottom_right = torch.ceil(j).long()
        i_bottom_right = torch.floor(i).long()
        j_bottom_right = torch.floor(j).long()
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]

        # Valid depth
        valid_depth = torch.min(
            torch.min(
                depth[i_top_left, j_top_left] > 0,
                depth[i_top_right, j_top_right] > 0
            ),
            torch.min(
                depth[i_bottom_left, j_bottom_left] > 0,
                depth[i_bottom_right, j_bottom_right] > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]
        # vaild index
        ids = ids.to(valid_depth.device)
        ids = ids[valid_depth]
        
        
        i = i[ids]
        j = j[ids]
        dist_i_top_left = i - i_top_left.double()
        dist_j_top_left = j - j_top_left.double()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left

        #depth is got from interpolation
        interpolated_depth = (
            w_top_left * depth[i_top_left, j_top_left] +
            w_top_right * depth[i_top_right, j_top_right] +
            w_bottom_left * depth[i_bottom_left, j_bottom_left] +
            w_bottom_right * depth[i_bottom_right, j_bottom_right]
        )

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]


    def read_valid_depth(self, mkpts1r, depth=None):
        depth = torch.tensor(depth).to(self.device)
        mkpts1r = mkpts1r.double().to(self.device)

        mkpts1r_a = torch.unsqueeze(mkpts1r[:,0],0)
        mkpts1r_b =  torch.unsqueeze(mkpts1r[:,1],0)
        mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0).to(self.device)

        depth, _, valid = self.interpolate_depth(mkpts1r_inter, depth)

        return depth, valid
    # def get_query_intrinsic(self, camera):
    #     """
    #     计算35mm等效焦距和内参矩阵。
        
    #     参数:
    #     image_width_px -- 图像的宽度（像素）
    #     image_height_px -- 图像的高度（像素）
    #     sensor_width_mm -- 相机传感器的宽度（毫米）
    #     sensor_height_mm -- 相机传感器的高度（毫米）
        
    #     返回:
    #     K -- 内参矩阵，形状为3x3
    #     """
    #     image_width_px, image_height_px, fx, fy ,cx, cy = camera
    #     # 计算内参矩阵中的焦距和主点坐标
        
    #     # 构建内参矩阵 K
    #     K = [[fx, 0, cx],
    #         [0, fy, cy],
    #         [0, 0, 1]]
        
    #     return K, image_width_px, image_height_px
    def get_intrinsic(self, camera):
        """
        计算35mm等效焦距和内参矩阵。
        
        参数:
        image_width_px -- 图像的宽度（像素）
        image_height_px -- 图像的高度（像素）
        sensor_width_mm -- 相机传感器的宽度（毫米）
        sensor_height_mm -- 相机传感器的高度（毫米）
        
        返回:
        K -- 内参矩阵，形状为3x3
        """
        if len(camera) == 5:
            image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
            # 计算内参矩阵中的焦距
            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y

            # 计算主点坐标
            cx = image_width_px / 2
            cy = image_height_px / 2
        elif len(camera) == 7:
            image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, f_mm = camera
            # 计算内参矩阵中的焦距
            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y
            
        elif len(camera) == 8:
            image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, fmm_x, fmm_y = camera
            # 计算内参矩阵中的焦距
            focal_ratio_x = fmm_x / sensor_width_mm
            focal_ratio_y = fmm_y / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y

        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        return K, image_width_px, image_height_px
    
    
    def get_query_intrinsic_single_focal(self, camera):
        """
        计算35mm等效焦距和内参矩阵。
        
        参数:
        image_width_px -- 图像的宽度（像素）
        image_height_px -- 图像的高度（像素）
        sensor_width_mm -- 相机传感器的宽度（毫米）
        sensor_height_mm -- 相机传感器的高度（毫米）
        
        返回:
        K -- 内参矩阵，形状为3x3
        """
        image_width_px, image_height_px, fmm, cx, cy = camera
        # 计算内参矩阵中的焦距和主点坐标
        fx, fy = fmm, fmm
        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        
        return K, image_width_px, image_height_px    


    def get_pose_mat(self, render_pose):
        """
        根据给定的渲染姿态计算并返回对应的变换矩阵。
        
        参数:
        - render_pose: 一个包含位置和姿态信息的列表或元组。
                    第一个元素是包含经度、纬度和高度的列表或元组。
                    第二个元素是包含欧拉角的列表，表示为俯仰、翻滚和偏航。
        
        返回:
        - T: 一个4x4的变换矩阵，用于将世界坐标系下的点转换到相机坐标系下。
        
        变换矩阵T由两部分组成：
        1. 旋转矩阵：将ENU坐标系下的点转换到ECEF坐标系下。
        2. 平移向量：将ECEF坐标系下的点平移到相机位置。
        
        示例:
        - render_pose = [
            [经度, 纬度, 高度],
            [俯仰, 翻滚, 偏航]
        ]
        - 返回的变换矩阵T可以用于将世界坐标系下的点转换到相机坐标系下。
        """
        # 提取经度、纬度和高度
        lon, lat, _ = render_pose[0]

        # 提取欧拉角并创建旋转矩阵
        euler_angles = render_pose[1].copy()
        rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
        rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天
        
        # 计算最终的旋转矩阵
        r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
        
        # 将经纬度转换为ECEF坐标系下的点
        xyz = WGS84_to_ECEF(render_pose[0])
        
        # 创建变换矩阵T，将旋转矩阵和平移向量合并
        T = np.concatenate((r, np.array([xyz]).transpose()), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)

        return T

    
    def enu_to_ned(self, points_enu):
        points_ned = np.zeros_like(points_enu)
        points_ned[:, 0] = points_enu[:, 0]  # East
        points_ned[:, 1] = points_enu[:, 1]  # North
        points_ned[:, 2] = -points_enu[:, 2]  # Up -> Down
        return points_ned
    def get_Points3D_torch(self, depth, R, t, K, points):
        """
        根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
        计算对应的三维世界坐标。

        参数:
        - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
        - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
        - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
        - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
        - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

        返回:
        - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
        """
        # 检查points是否为同质坐标，如果不是则扩展为同质坐标
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
            points_2D = points_2D.T
        else:
            points_2D = points.T

        # 扩展平移向量以匹配点的数量
        t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
        t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

        # 将所有输入转换为高精度浮点数类型
        points_2D = points_2D.float()
        K = K.float()
        R = R.float()
        depth = depth.float()
        t = t.float()

        # 修改内参矩阵的最后一项，以适应透视投影
        K[-1, -1] = -1

        # 计算三维世界坐标
        Points_3D = R @ (K @ (depth * points_2D)) + t
        # 返回三维点坐标，形状为 [n, 3]
        return Points_3D.cpu().numpy().T
    def get_Points3D_torch_v2(self, depth, R, t, K, points):
        """
        根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
        计算对应的三维世界坐标。

        参数:
        - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
        - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
        - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
        - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
        - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

        返回:
        - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
        """
        # 检查points是否为同质坐标，如果不是则扩展为同质坐标
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
            points_2D = points_2D.T
        else:
            points_2D = points.T

        # 扩展平移向量以匹配点的数量
        t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
        t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

        # 将所有输入转换为高精度浮点数类型
        points_2D = points_2D.float()
        K = K.float()
        R = R.float()
        depth = depth.float()
        t = t.float()

        # 计算三维世界坐标
        Points_3D = R @ (K @ (depth * points_2D)) + t
        # 返回三维点坐标，形状为 [n, 3]
        return Points_3D.cpu().numpy().T
    def get_Points3D(self, depth, R, t, K, points):
        """
        根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
        计算对应的三维世界坐标。

        参数:
        - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
        - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
        - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
        - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
        - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

        返回:
        - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
        """
        # 检查points是否为同质坐标，如果不是则扩展为同质坐标
        if points.shape[-1] != 3:
            points_2D = np.concatenate([points, np.ones_like(points[ :, [0]])], axis=-1)
            points_2D = points_2D.T
        else:
            points_2D = points.T  # 确保points的形状为 [2, n]

        # 扩展平移向量以匹配点的数量
        
        t = np.expand_dims(t,-1)
        t = np.tile(t, points_2D.shape[-1])

        # 将所有输入转换为高精度浮点数类型
        points_2D = np.float64(points_2D)
        K = np.float64(K)
        R = np.float64(R)
        depth = np.float64(depth)
        t = np.float64(t)

        # 修改内参矩阵的最后一项，以适应透视投影
        K[-1, -1] = -1
        
        # 计算三维世界坐标
        Points_3D = R @ K @ (depth * points_2D) + t
        # 返回三维点坐标，形状为 [3, n]
        return Points_3D.T
    def get_new_focal_lens(self, R, t, K, points_3D, points):
        # 将输入数据转换为高精度浮点数类型
        x, y = points
        points_3D = np.float64(points_3D)
        K = np.float64(K)
        R = np.float64(R)
        t = np.float64(t)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R_inverse = np.linalg.inv(R)
        point_3d_camera = np.expand_dims(points_3D - t, 1)
        # 将世界坐标系下的点转换为相机坐标系下的点
        point_3d_camera_r = R_inverse @ point_3d_camera
        Rx, Ry, Rz = point_3d_camera_r
        
        fx_new = (x - cx) * Rz / Rx
        fy_new = (y - cy) * Rz / Ry
        print("--------------------------------")
        print("Rx, Ry, Rz", Rx, Ry, Rz )
        print("fx_new, fy_new", fx_new, fy_new)
    def get_points2D_torch(self, R, t, K, points_3D):  # points_3D[n,3]
        """
        根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及三维世界坐标，
        计算对应的二维图像坐标。

        参数:
        - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
        - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
        - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
        - points_3D: 三维世界坐标数组，尺寸为 [n, 3]，其中 n 是点的数量。
        返回:
        - point_2d: 二维图像坐标数组，尺寸为 [n, 2]。
        """
        # 将输入数据转换为PyTorch张量
        points_3D = torch.tensor(points_3D, dtype=torch.float64)
        K = torch.tensor(K, dtype=torch.float64)
        R = torch.tensor(R, dtype=torch.float64)
        t = torch.tensor(t, dtype=torch.float64)
        t = t.unsqueeze(1)
        t = t.repeat(1, points_3D.size(0)).transpose(0, 1)

        # 修改内参矩阵的最后一项，以适应透视投影
        K[-1, -1] = -1

        # 计算相机坐标系下的点
        point_3d_camera = points_3D - t
        # 将世界坐标系下的点转换为相机坐标系下的点
        point_3d_camera_r = torch.matmul(R.inverse(), point_3d_camera.transpose(0, 1))
        # 将相机坐标系下的点投影到图像平面，得到同质坐标
        point_2d_homo = torch.matmul(K.inverse(), torch.cat((point_3d_camera_r, torch.ones(points_3D.shape[0], 1, device=points_3D.device)), dim=1))
        # 将同质坐标转换为二维图像坐标
        point_2d = point_2d_homo[:, :2] / point_2d_homo[:, 2:]
        return point_2d        
    def get_points2D_ECEF(self, R, t, K, points_3D):  # points_3D[n,3]
        """
        根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及三维世界坐标，
        计算对应的二维图像坐标。

        参数:
        - R: 旋转矩阵，尺寸为 [3, 3]，表示从相机坐标系到世界坐标系的旋转。
        - t: 平移向量，尺寸为 [3, 1]，表示从相机坐标系到世界坐标系的平移。
        - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
        - points_3D: 三维世界坐标数组，尺寸为 [n, 3]，其中 n 是点的数量。
        返回:
        - point_2d: 二维图像坐标数组，尺寸为 [n, 2]。
        """
        # 将输入数据转换为高精度浮点数类型
        points_3D = np.float64(points_3D)
        K = np.float64(K)
        R = np.float64(R)
        t = np.float64(t)
        # 修改内参矩阵的最后一项，以适应透视投影
        K[-1, -1] = -1
        
        K_inverse = np.linalg.inv(K)
        R_inverse = np.linalg.inv(R)
        # 计算相机坐标系下的点
        point_3d_camera = np.expand_dims(points_3D - t, 1)
        # 将世界坐标系下的点转换为相机坐标系下的点
        point_3d_camera_r = R_inverse @ point_3d_camera
        # 将相机坐标系下的点投影到图像平面，得到同质坐标
        point_2d_homo = K_inverse @ point_3d_camera_r
        # 将同质坐标转换为二维图像坐标
        point_2d = point_2d_homo / point_2d_homo[2]
        return point_2d.T
    def get_points2D_CGCS2000(self, R, t, K, points_3D):  # points_3D[n,3]
        """
        根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及三维世界坐标，
        计算对应的二维图像坐标。

        参数:
        - R: 旋转矩阵，尺寸为 [3, 3]，表示从相机坐标系到世界坐标系的旋转。
        - t: 平移向量，尺寸为 [3, 1]，表示从相机坐标系到世界坐标系的平移。
        - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
        - points_3D: 三维世界坐标数组，尺寸为 [n, 3]，其中 n 是点的数量。
        返回:
        - point_2d: 二维图像坐标数组，尺寸为 [n, 2]。
        """
        # 将输入数据转换为高精度浮点数类型
        points_3D = np.float64(points_3D)
        K = np.float64(K)
        R = np.float64(R)
        t = np.float64(t)
        # 修改内参矩阵的最后一项，以适应透视投影
        # K[-1, -1] = -1
        
        K_inverse = np.linalg.inv(K)
        R_inverse = np.linalg.inv(R)
        # 计算相机坐标系下的点
        point_3d_camera = np.expand_dims(points_3D - t, 1)
        # 将世界坐标系下的点转换为相机坐标系下的点
        point_3d_camera_r = R_inverse @ point_3d_camera
        # 将相机坐标系下的点投影到图像平面，得到同质坐标
        point_2d_homo = K_inverse @ point_3d_camera_r
        # 将同质坐标转换为二维图像坐标
        point_2d = point_2d_homo / point_2d_homo[2]
        return point_2d.T
    def localize_using_pnp_opencv(self, points3D, points2D, query_camera):
        """
        使用 OpenCV 的 PnP 算法估计相机位姿。
        
        参数:
        - points3D: 三维世界坐标点的数组，尺寸为 [n, 3]，其中 n 是点的数量。
        - points2D: 对应的二维图像坐标点的数组，尺寸为 [n, 2]。
        - query_camera: 相机的内参矩阵，尺寸为 [3, 3]。
        
        返回:
        - ret: 一个字典，包含位姿估计的成功标志、旋转矩阵和平移向量。
        
        注意：points3D 和 points2D 中的坐标点应该以相同的顺序对应。
        """
        distortion_coeffs = np.zeros((4, 1))
        success, vector_rotation, vector_translation = cv2.solvePnP(
                                                                -points3D, 
                                                                -points2D, 
                                                                query_camera, 
                                                                distortion_coeffs, 
                                                                flags=cv2.SOLVEPNP_EPNP
        )
                                                                
        rot_mat, _ = cv2.Rodrigues(vector_rotation)

        R_c2w = np.asmatrix(rot_mat).transpose()
        

        t_c2w = -R_c2w.dot(vector_translation)
        

        ret = {}
        ret['success'] = success
        ret['qvec'] = rot_mat.transpose()  #! 旋转矩阵转置
        ret['tvec'] = np.array(-t_c2w).reshape(3)  #! 平移向量取负数，并重塑为一维数组
        return ret

    def localize_using_pnp(self, points3D, points2D, query_camera, width, height):
        points3D = [points3D[i] for i in range(points3D.shape[0])]
        fx, fy, cx, cy = query_camera[0][0], query_camera[1][1], query_camera[0][2], query_camera[1][2]
        cfg = {
            "model": "PINHOLE",
            "width": width,
            "height": height,
            "params": [fx, fy, cx, cy],
        }  

        ret = pycolmap.absolute_pose_estimation(
            points2D,
            points3D,
            cfg,
            estimation_options=self.config.get('estimation', {}),
            refinement_options=self.config.get('refinement', {}),
        )

        return ret  
        
    def get_target_location(self, target_point, render_pose, depth_mat, query_camera):
        # target_point = [[960, 540]]
        
        target_point = torch.tensor(target_point)
        render_K, _, render_height_px = self.get_intrinsic(query_camera) # 将depth图resize到查询图尺寸，因此内参一致
        
        render_K = np.float64(render_K)
        
        K_c2w =  np.linalg.inv(render_K)
        # Get render pose
        render_T = np.float64(self.get_pose_mat(render_pose))  #! 旋转矩阵转换到ECEF坐标系下
        print('target point: ', target_point)
        depth, _ = self.read_valid_depth(target_point, depth = depth_mat)
        
        # target_point[:, 1] = render_height_px - target_point[:, 1] # target_point height 修正
        # # 
        # # Compute 3D points
        # render_K[1,2] = render_height_px - render_K[1,2]
        # Points_3D = self.get_Points3D(
        #     depth,
        #     render_T[:3, :3],
        #     render_T[:3, 3],
        #     np.linalg.inv(render_K),
        #     target_point.clone().detach(),
        # )
        
        
        # target_point[:, 1] = render_height_px - target_point[:, 1]
        render_T[:3, 1] = -render_T[:3, 1] 
        render_T[:3, 2] = -render_T[:3, 2] 
        Points_3D = self.get_Points3D_torch_v2(
            depth,
            torch.tensor(render_T[:3, :3]).to(self.device) ,
            torch.tensor(render_T[:3, 3]).to(self.device) ,
            torch.tensor(K_c2w).to(self.device) ,
            target_point.to(self.device) 
        )
        
        # Points_3D = Points_3D.squeeze(0)
        ECEF_res = [Points_3D[:,0], Points_3D[:,1], Points_3D[:,2]]
        
        lon, lat, height = ECEF_to_WGS84(ECEF_res)  #! ECEF_to_WGS84
        n_points = len(lon)
        wgs84_res = np.zeros((n_points, 3), dtype=np.float64)
        ECEF_res = np.zeros((n_points, 3), dtype=np.float64)
        for i in range(n_points):
            wgs84_res[i][0] = lon[i]
            wgs84_res[i][1] = lat[i]
            wgs84_res[i][2] = height[i]

            ECEF_res[i][0] = Points_3D[i][0]
            ECEF_res[i][1] = Points_3D[i][1]
            ECEF_res[i][2] = Points_3D[i][2]
        return ECEF_res, wgs84_res
    
  
    def estimate_query_pose(self, matches, render_pose, depth_mat, query_camera=None, render_camera = None, homo = None):
        """
        Main function to perform localization on query images.

        Args:
            config (dict): Configuration settings.
            data (dict): Data related to queries and renders.
            iter (int): Iteration number for naming output files.
            outputs (Path): Path to the output directory.
            con (dict, optional): Additional configuration for the localization process.

        Returns:
            Path: Path to the saved estimated pose file.
        """
        # Get render intrinsics and query intrinsics
        query_K, query_width_px, query_height_px = self.get_intrinsic(query_camera)

        render_K, _, render_height_px = self.get_intrinsic(render_camera)
        render_K = torch.tensor(render_K, device=self.device)
        K_c2w = render_K.inverse()
        # print('----renderK', render_K)
        # print('----queryK', query_K)
        # Get render pose
        render_T = torch.tensor(self.get_pose_mat(render_pose), device=self.device)  #! 旋转矩阵转换到ECEF坐标系下
        
        # Get 2D-2D matches
        mkpts_q, mkpts_r = matches
        depth, valid = self.read_valid_depth(mkpts_r, depth = depth_mat)
        # Compute 3D points
        mkpts_r_in_osg = mkpts_r.clone().detach().to(self.device) 
        mkpts_r_in_osg[:, 1] = render_height_px - mkpts_r_in_osg[:, 1]
        render_K[1, 2] = render_height_px - render_K[1,2]
        Points_3D = self.get_Points3D_torch(
            depth,
            render_T[:3, :3],
            render_T[:3, 3],
            render_K.inverse(),
            mkpts_r_in_osg.clone().detach(),
        )
        
        # render_T[:3, 1] = -render_T[:3, 1] 
        # render_T[:3, 2] = -render_T[:3, 2] 
        # Points_3D_v2 = self.get_Points3D_torch_v2(
        #     depth,
        #     render_T[:3, :3],
        #     render_T[:3, 3],
        #     K_c2w,
        #     mkpts_r.to(self.device) 
            
        # )
        valid = valid.to(mkpts_q.device)
        query_K = np.array(query_K)
        # query_K[1, 2] = query_height_px - query_K[1, 2]
        points2D_v2 = mkpts_q[valid].cpu().numpy()
        ret_v2 = self.localize_using_pnp(Points_3D, points2D_v2, query_K, query_width_px, query_height_px)
        # logger.info('Starting localization...')
        # Perform PnP to find camera pose
        
        # mkpts_q_in_osg = mkpts_q.clone().detach() 
        # mkpts_q_in_osg[:, 1] = query_height_px - mkpts_q_in_osg[:, 1]
        # points2D = mkpts_q_in_osg[valid].cpu().numpy()
        # query_K[1, 2] = query_height_px - query_K[1,2]
        # query_K_inv_osg = np.linalg.inv(copy.deepcopy(query_K))
        # query_K_inv_osg[-1, -1] = -1  # K_c2w
        # ret = self.localize_using_pnp_opencv(Points_3D, points2D, np.linalg.inv(query_K_inv_osg))

        if 'qvec' in ret_v2:
            q_w2c = ret_v2['qvec']
            t_w2c = ret_v2['tvec']
            R_c2w = np.asmatrix(qvec2rotmat(q_w2c)).transpose()  
            t_c2w = np.array(-R_c2w.dot(t_w2c))  
            R_c2w[:3, 1] = -R_c2w[:3, 1]
            R_c2w[:3, 2] = -R_c2w[:3, 2]
            ret_v2['qvec'] = R_c2w
            ret_v2['tvec'] = t_c2w[0]
            if homo:
                return ret_v2, Points_3D, mkpts_q[valid].cpu().numpy()
        else:
            if homo:
                return None, None, None
            else: 
                return None
        return ret_v2
    def homo_pnp(self, Points_3D, points2D, query_camera):
        query_K, _, query_height_px = self.get_intrinsic(query_camera)
        
        mkpts_q_in_osg = copy.deepcopy(points2D)
        mkpts_q_in_osg[:, 1] = query_height_px - mkpts_q_in_osg[:, 1]
        
        
        query_K_inv_osg = np.linalg.inv(copy.deepcopy(query_K))
        query_K_inv_osg[-1, -1] = -1
        ret = self.localize_using_pnp_opencv(Points_3D, mkpts_q_in_osg, np.linalg.inv(query_K_inv_osg))
        return ret
    def find_homography(self, matches, color_img, name):
        query_point = matches[0]
        reference_point = matches[1]
        H,  _ = cv2.findHomography(query_point.numpy(), reference_point.numpy(), cv2.RANSAC, 5.0)
        query_img = cv2.imread(self.query_path)
        # 可视化验证图片
        h_ori_query = cv2.warpPerspective(query_img, H, (query_img.shape[1], query_img.shape[0]))
        alpha = 0.5
        h_ori_query[:, :, 2]  = (1 - alpha) * color_img[:, :, 2] + alpha * h_ori_query[:, :, 2]
        combined_image = cv2.addWeighted(color_img, 1, h_ori_query, alpha, 0)
        temp_name = name + '.jpg'
        if not os.path.exists(self.outputs/"combited_image"):
            os.mkdir(self.outputs/"combited_image")
        path = str(self.outputs/"combited_image"/temp_name)
        cv2.imwrite(path, combined_image)
        
        return H
def vis_points(points_3d):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # 提取x, y, z坐标
    x_coords = [point[0] for point in points_3d]
    y_coords = [point[1] for point in points_3d]
    z_coords = [point[2] for point in points_3d]

    # 创建一个新的图形
    fig = plt.figure()

    # 添加一个3D轴
    ax = fig.add_subplot(111, projection='3d')

    # 在3D轴上散点图
    ax.scatter(x_coords, y_coords, z_coords)

    # 设置轴标签
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # 显示图形
    plt.show()
def vis_map(point3d, point2d):
    import matplotlib.pyplot as plt
    import mplcursors

    # 创建一张空白图像
    image_path = '/home/ubuntu/Documents/code/github/target_indicator_server/fast_render2loc/datasets/松兰山/prior_images/Z_0_42.png'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB
    # 创建matplotlib图像
    fig, ax = plt.subplots()
    img_plot = ax.imshow(image_rgb) 
    points = {}
    # 定义一些像素坐标和相关信息

    for i in range(len(point3d)):
        x = int(point2d[i][0])
        y = int(point2d[i][1])
        info = str(point3d[i])
        points[(x, y)] = info
    # 定义鼠标悬停时的回调函数
    def on_hover(sel):
        # 将matplotlib坐标转换为OpenCV坐标（y, x）
        y, x = sel.target.index
        print("xy", (x, y))
        if (x, y) in points:
            sel.annotation.set_text(points[(x, y)])
        else:
            sel.annotation.set_text('No info')
    # 使用mplcursors添加鼠标悬停功能
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", on_hover)

    # 显示图像
    plt.show()
    
def get_Points3D_torch_v2(depth, R, t, K, points):
        """
        根据相机的内参矩阵、姿态（旋转矩阵和平移向量）以及图像上的二维点坐标和深度信息，
        计算对应的三维世界坐标。

        参数:
        - depth: 深度值数组，尺寸为 [n,]，其中 n 是点的数量。
        - R: 旋转矩阵，尺寸为 [3, 3]，表示从世界坐标系到相机坐标系的旋转。
        - t: 平移向量，尺寸为 [3, 1]，表示从世界坐标系到相机坐标系的平移。
        - K: 相机内参矩阵，尺寸为 [3, 3]，包含焦距和主点坐标。
        - points: 二维图像坐标数组，尺寸为 [n, 2]，其中 n 是点的数量。

        返回:
        - Points_3D: 三维世界坐标数组，尺寸为 [n, 3]。
        """
        # 检查points是否为同质坐标，如果不是则扩展为同质坐标
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
            points_2D = points_2D.T
        else:
            points_2D = points.T

        # 扩展平移向量以匹配点的数量
        t = t.unsqueeze(1)  # 这相当于np.expand_dims(t, -1)
        t = t.repeat(1, points_2D.size(-1))  # 这相当于np.tile(t, points_2D.shape[-1])

        # 将所有输入转换为高精度浮点数类型
        points_2D = points_2D.float()
        K = K.float()
        R = R.float()
        depth = depth.float()
        t = t.float()

        # 计算三维世界坐标
        Points_3D = R @ (K @ (depth * points_2D)) + t
        # 返回三维点坐标，形状为 [n, 3]
        return Points_3D.cpu().numpy().T
def get_pose_mat(render_pose):
        """
        根据给定的渲染姿态计算并返回对应的变换矩阵。
        
        参数:
        - render_pose: 一个包含位置和姿态信息的列表或元组。
                    第一个元素是包含经度、纬度和高度的列表或元组。
                    第二个元素是包含欧拉角的列表，表示为俯仰、翻滚和偏航。
        
        返回:
        - T: 一个4x4的变换矩阵，用于将世界坐标系下的点转换到相机坐标系下。
        
        变换矩阵T由两部分组成：
        1. 旋转矩阵：将ENU坐标系下的点转换到ECEF坐标系下。
        2. 平移向量：将ECEF坐标系下的点平移到相机位置。
        
        示例:
        - render_pose = [
            [经度, 纬度, 高度],
            [俯仰, 翻滚, 偏航]
        ]
        - 返回的变换矩阵T可以用于将世界坐标系下的点转换到相机坐标系下。
        """
        # 提取经度、纬度和高度
        lon, lat, _ = render_pose[0]

        # 提取欧拉角并创建旋转矩阵
        euler_angles = render_pose[1].copy()
        rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
        rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()  # ZXY 东北天
        
        # 计算最终的旋转矩阵
        r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
        
        # 将经纬度转换为ECEF坐标系下的点
        xyz = WGS84_to_ECEF(render_pose[0])
        
        # 创建变换矩阵T，将旋转矩阵和平移向量合并
        T = np.concatenate((r, np.array([xyz]).transpose()), axis=1)
        T = np.concatenate((T, np.array([[0, 0, 0, 1]])), axis=0)

        return T

    
def get_intrinsic(camera):
        """
        计算35mm等效焦距和内参矩阵。
        
        参数:
        image_width_px -- 图像的宽度（像素）
        image_height_px -- 图像的高度（像素）
        sensor_width_mm -- 相机传感器的宽度（毫米）
        sensor_height_mm -- 相机传感器的高度（毫米）
        
        返回:
        K -- 内参矩阵，形状为3x3
        """
        if len(camera) == 5:
            image_width_px, image_height_px, sensor_width_mm, sensor_height_mm, f_mm = camera
            # 计算内参矩阵中的焦距
            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y

            # 计算主点坐标
            cx = image_width_px / 2
            cy = image_height_px / 2
        elif len(camera) == 7:
            image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, f_mm = camera
            # 计算内参矩阵中的焦距
            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y
            
        elif len(camera) == 8:
            image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, fmm_x, fmm_y = camera
            # 计算内参矩阵中的焦距
            focal_ratio_x = fmm_x / sensor_width_mm
            focal_ratio_y = fmm_y / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y

        # 构建内参矩阵 K
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        return K, image_width_px, image_height_px
def read_valid_depth(mkpts1r, depth=None):
    depth = torch.tensor(depth).cuda()
    mkpts1r = mkpts1r.double().cuda()

    mkpts1r_a = torch.unsqueeze(mkpts1r[:,0],0)
    mkpts1r_b =  torch.unsqueeze(mkpts1r[:,1],0)
    mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0).cuda()

    depth, _, valid = interpolate_depth(mkpts1r_inter, depth)

    return depth, valid
def interpolate_depth(pos, depth):
        ids = torch.arange(0, pos.shape[0])
        if depth.ndim != 2:
            if depth.ndim == 3:
                depth = depth[:,:,0]
            else:
                raise Exception("Invalid depth image!")
        h, w = depth.size()
        
        i = pos[:, 0]
        j = pos[:, 1]

        # Valid corners, check whether it is out of range
        i_top_left = torch.floor(i).long()
        j_top_left = torch.floor(j).long()
        valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

        i_top_right = torch.floor(i).long()
        # j_top_right = torch.ceil(j).long()
        j_top_right = torch.floor(j).long()
        valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

        # i_bottom_left = torch.ceil(i).long()
        i_bottom_left = torch.floor(i).long()
        
        j_bottom_left = torch.floor(j).long()
        valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

        # i_bottom_right = torch.ceil(i).long()
        # j_bottom_right = torch.ceil(j).long()
        i_bottom_right = torch.floor(i).long()
        j_bottom_right = torch.floor(j).long()
        valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

        valid_corners = torch.min(
            torch.min(valid_top_left, valid_top_right),
            torch.min(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corners]
        j_top_left = j_top_left[valid_corners]

        i_top_right = i_top_right[valid_corners]
        j_top_right = j_top_right[valid_corners]

        i_bottom_left = i_bottom_left[valid_corners]
        j_bottom_left = j_bottom_left[valid_corners]

        i_bottom_right = i_bottom_right[valid_corners]
        j_bottom_right = j_bottom_right[valid_corners]

        # Valid depth
        valid_depth = torch.min(
            torch.min(
                depth[i_top_left, j_top_left] > 0,
                depth[i_top_right, j_top_right] > 0
            ),
            torch.min(
                depth[i_bottom_left, j_bottom_left] > 0,
                depth[i_bottom_right, j_bottom_right] > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]
        # vaild index
        ids = ids.to(valid_depth.device)
        ids = ids[valid_depth]
        
        
        i = i[ids]
        j = j[ids]
        dist_i_top_left = i - i_top_left.double()
        dist_j_top_left = j - j_top_left.double()
        w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
        w_top_right = (1 - dist_i_top_left) * dist_j_top_left
        w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
        w_bottom_right = dist_i_top_left * dist_j_top_left

        #depth is got from interpolation
        interpolated_depth = (
            w_top_left * depth[i_top_left, j_top_left] +
            w_top_right * depth[i_top_right, j_top_right] +
            w_bottom_left * depth[i_bottom_left, j_bottom_left] +
            w_bottom_right * depth[i_bottom_right, j_bottom_right]
        )

        pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

        return [interpolated_depth, pos, ids]
def read_pfm(file):
    """ 读取 PFM 文件并返回 (image, scale) """
    with open(file, 'rb') as f:
        header = f.readline().decode('utf-8').strip()
        if header not in ['PF', 'Pf']:
            raise Exception('Not a PFM file.')
        dim_line = f.readline().decode('utf-8').strip()
        width, height = map(int, dim_line.split())
        scale = float(f.readline().decode('utf-8').strip())
        color = (header == 'PF')
        data = np.fromfile(f, dtype=np.float32)
        if color:
            image = np.reshape(data, (height, width, 3))
        else:
            image = np.reshape(data, (height, width))
        return image, scale
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# ---------- 工具：均匀网格采样 ----------
def grid_sample_xy(width, height, stride=4, offset=0.5, return_index=False):
    """
    生成均匀采样网格像素坐标 (u, v)，u=列(x), v=行(y)。
    offset=0.5 表示以像素中心为射线起点，避免偏置。
    """
    us = np.arange(0, width,  stride, dtype=np.float64) + offset
    vs = np.arange(0, height, stride, dtype=np.float64) + offset
    uu, vv = np.meshgrid(us, vs)  # shape [H/str, W/str]
    u = uu.reshape(-1)
    v = vv.reshape(-1)
    if return_index:
        iu = (u - offset).astype(np.int32)
        iv = (v - offset).astype(np.int32)
        return u, v, iu, iv
    return u, v

# ---------- 工具：反投影 ----------
def backproject_depth_to_ecef(
    depth_m: np.ndarray,             # [H,W], 单位米；无效=0
    K_c2w: np.ndarray,                   # 3x3 内参
    render_T,
    stride: int = 4,
    d_min: float = 0.1,
    d_max: float = 64000.0,
    rgb: np.ndarray | None = None,   # [H,W,3] (BGR或RGB都可，下面会统一成RGB)
):
    """
    返回：
      pts_ecef: [N,3]  反投影后的 ECEF 点
      cols_rgb: [N,3]  (0-255) 颜色（如未提供 rgb，则返回 None）
      mask_idx: [N]    采样像素的 (v,u) 索引，用于后处理
    """
    # 1) 生成均匀采样像素点 (u, v)
    H, W = depth_m.shape[:2]

    def grid_sample_xy(width, height, stride=4, offset=0.5, return_index=False):
        us = np.arange(0, width,  stride, dtype=np.float64) + offset
        vs = np.arange(0, height, stride, dtype=np.float64) + offset
        uu, vv = np.meshgrid(us, vs)  # [H/str, W/str]
        u = uu.reshape(-1); v = vv.reshape(-1)
        if return_index:
            iu = (u - offset).astype(np.int32)
            iv = (v - offset).astype(np.int32)
            return u, v, iu, iv
        return u, v

    u, v, iu, iv = grid_sample_xy(W, H, stride=2, offset=0.5, return_index=True)

    # 2) 取得这些像素位置的深度（最近邻）

    # 可选：做有效性筛选（>0 且不过大）

    # 3) 组装 points[N,2] 并调用反投影
    points_np = np.stack([u, v], axis=1)  # [N,2]
    depth_vec, _ = read_valid_depth(torch.tensor(points_np), depth = depth_mat)
    depth_vec = depth_vec.cpu().numpy()
    valid = np.isfinite(depth_vec) & (depth_vec > 0.1) & (depth_vec <= 64000.0)
    u, v, depth_vec = u[valid], v[valid], depth_vec[valid]

    Points_3D = get_Points3D_torch_v2(
        torch.tensor(depth_vec, dtype=torch.float32, device="cuda"),
        torch.tensor(render_T[:3, :3], dtype=torch.float32, device="cuda"),
        torch.tensor(render_T[:3, 3],  dtype=torch.float32, device="cuda"),
        torch.tensor(K_c2w,            dtype=torch.float32, device="cuda"),  # K^{-1}
        torch.tensor(points_np[valid],        dtype=torch.float32, device="cuda"),
    )
    # Points_3D: [N,3] (numpy)
    # 取颜色
    cols = None
    if rgb is not None:
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            # 统一为 RGB
            if rgb.dtype != np.uint8:
                rgb_show = np.clip(rgb, 0, 255).astype(np.uint8)
            else:
                rgb_show = rgb
            # 如果是 OpenCV 读入的 BGR，改为 RGB
            # 你可以按自己数据来源选择是否转换
            if isinstance(rgb, np.ndarray) and rgb.flags['C_CONTIGUOUS']:
                # 假定来自 cv2.imread -> BGR
                rgb_show = cv2.cvtColor(rgb_show, cv2.COLOR_BGR2RGB)
            cols = rgb_show[iv, iu, :]              # [N,3]
        else:
            print("[warn] rgb shape should be HxWx3, ignore colors.")
    return Points_3D, cols, (iv, iu)

# ---------- 可视化 ----------
def quick_scatter3d(points_xyz: np.ndarray, colors_rgb: np.ndarray | None = None, max_points=200000):
    """
    轻量 3D 散点可视化（matplotlib）。点数过大时随机下采样。
    """
    if points_xyz.shape[0] == 0:
        print("[info] no valid 3D points to visualize.")
        return
    idx = np.arange(points_xyz.shape[0])
    if points_xyz.shape[0] > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)
    P = points_xyz[idx]
    C = None if colors_rgb is None else (colors_rgb[idx] / 255.0)

    fig = plt.figure(figsize=(7,6), dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P[:,0], P[:,1], P[:,2], s=0.2, c=C)
    ax.set_xlabel("X (ECEF)"); ax.set_ylabel("Y (ECEF)"); ax.set_zlabel("Z (ECEF)")
    ax.set_title("Dense Back-Projected Point Cloud (ECEF)")
    plt.tight_layout(); plt.show()

# ---------- 保存 PLY ----------
def save_ply(path: str | Path, pts_xyz: np.ndarray, cols_rgb: np.ndarray | None = None):
    path = Path(path)
    N = pts_xyz.shape[0]
    has_color = cols_rgb is not None and cols_rgb.shape[0] == N
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_color:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_color:
            for (x,y,z), (r,g,b) in zip(pts_xyz, cols_rgb.astype(np.uint8)):
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for (x,y,z) in pts_xyz:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    print(f"[ply] saved -> {path.resolve()}")
def get_init(pose_file, start = 0):
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # if '14000' in parts[0]:
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                # pitch, roll, yaw, lon, lat, alt,  = map(float, parts[1: ])
                
                euler_angles = [pitch, roll, yaw]
                translation = [lon, lat, alt]
                origin = WGS84_to_ECEF(translation)
                if start == int(parts[0].split('_')[0]):
                    break
    return euler_angles, translation, origin
def quick_scatter3d_minimal(points_xyz: np.ndarray,
                            colors_rgb: np.ndarray | None = None,
                            max_points=200_000,
                            save_path: str | None = None,
                            project_planes=("xy",),   # 可选 {"xy","yz","xz"} 的子集
                            shadow_alpha=0.15,
                            shadow_size=0.15):
    """
    极简黑色坐标轴 + 右下角三轴图标 + 彩色点云
    新增：在指定平面上做彩色“投影阴影”，营造立体感。
      project_planes: ("xy",) 只地面；也可 ("xy","yz","xz") 三面
      shadow_alpha:   阴影透明度
      shadow_size:    阴影点尺寸（相对原始点的比例）
    """
    if points_xyz.shape[0] == 0:
        print("[info] no valid 3D points to visualize.")
        return

    # 采样
    idx = np.arange(points_xyz.shape[0])
    if points_xyz.shape[0] > max_points:
        idx = np.random.choice(idx, size=max_points, replace=False)

    P = points_xyz[idx]
    C = None if colors_rgb is None else (colors_rgb[idx] / 255.0)

    # 透明背景
    fig = plt.figure(figsize=(7, 6), dpi=160)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('none')
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor('none')
        axis.pane.fill = False

    # 黑色轴线 + 极简刻度
    ax.w_xaxis.line.set_color("black")
    ax.w_yaxis.line.set_color("black")
    ax.w_zaxis.line.set_color("black")
    ax.tick_params(colors="black", labelsize=8)

    # 点云主体
    main_size = 0.25
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=main_size, c=C, depthshade=False)

    # 等比例包围盒
    xyz_min, xyz_max = P.min(axis=0), P.max(axis=0)
    center = (xyz_min + xyz_max) / 2.0
    span = (xyz_max - xyz_min).max()
    r = span * 0.55 if span > 0 else 1.0
    ax.set_xlim(center[0]-r, center[0]+r)
    ax.set_ylim(center[1]-r, center[1]+r)
    ax.set_zlim(center[2]-r, center[2]+r)

    # 中心黑色三轴
    ax.add_line(Line3D([center[0]-r, center[0]+r], [center[1], center[1]], [center[2], center[2]], color='black', linewidth=1.0))
    ax.add_line(Line3D([center[0], center[0]], [center[1]-r, center[1]+r], [center[2], center[2]], color='black', linewidth=1.0))
    ax.add_line(Line3D([center[0], center[0]], [center[1], center[1]], [center[2]-r, center[2]+r], color='black', linewidth=1.0))
    ax.set_xlabel('X', color='black', labelpad=6)
    ax.set_ylabel('Y', color='black', labelpad=6)
    ax.set_zlabel('Z', color='black', labelpad=6)

    # —— 彩色“投影阴影”（把点投到指定平面上，用较小点+透明度显示）——
    if C is None:
        # 若没有颜色，给个灰色投影
        C_shadow = [(.2, .2, .2)] * P.shape[0]
    else:
        C_shadow = C

    s_shadow = main_size * (shadow_size / max(1e-6, main_size))
    if "xy" in project_planes:
        z0 = xyz_min[2]  # 地面
        ax.scatter(P[:, 0], P[:, 1], np.full_like(P[:, 2], z0),
                   s=s_shadow, c=C_shadow, depthshade=False, alpha=shadow_alpha)

    if "yz" in project_planes:
        x0 = xyz_min[0]  # 左侧墙
        ax.scatter(np.full_like(P[:, 0], x0), P[:, 1], P[:, 2],
                   s=s_shadow, c=C_shadow, depthshade=False, alpha=shadow_alpha)

    if "xz" in project_planes:
        y0 = xyz_min[1]  # 后侧墙
        ax.scatter(P[:, 0], np.full_like(P[:, 1], y0), P[:, 2],
                   s=s_shadow, c=C_shadow, depthshade=False, alpha=shadow_alpha)

    # —— 右下角三色三轴小图标 ——（保留原样）
    inset = fig.add_axes([0.83, 0.08, 0.12, 0.12])
    inset.set_facecolor('none')
    inset.set_xlim(0, 1); inset.set_ylim(0, 1)
    inset.axis('off')
    o = np.array([0.15, 0.15])   # 原点
    L = 0.65
    inset.arrow(o[0], o[1], L, 0, width=0.012, head_width=0.06, head_length=0.08,
                length_includes_head=True, color='#E53935')
    inset.text(o[0] + L + 0.05, o[1], 'X', color='#E53935', fontsize=9, va='center')
    inset.arrow(o[0], o[1], 0, L, width=0.012, head_width=0.06, head_length=0.08,
                length_includes_head=True, color='#43A047')
    inset.text(o[0], o[1] + L + 0.05, 'Y', color='#43A047', fontsize=9, ha='center')
    inset.arrow(o[0], o[1], L*0.6, L*0.6, width=0.012, head_width=0.06, head_length=0.08,
                length_includes_head=True, color='#1E88E5')
    inset.text(o[0] + L*0.6 + 0.05, o[1] + L*0.6 + 0.02, 'Z', color='#1E88E5', fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches='tight', transparent=True)
        print(f"[saved] {save_path}")
    plt.show()
from pyproj import CRS, Transformer
import math
def get_utm_epsg_from_lonlat(lon, lat):
    """
    根据经纬度 (lon, lat) 计算其对应的 UTM 分带 EPSG 号。
    - 北半球：EPSG:326XX
    - 南半球：EPSG:327XX

    :param lon: 经度 (float)，通常范围 [-180, 180]
    :param lat: 纬度 (float)，通常范围 [-90, 90]
    :return: 整型 EPSG，若无法计算则返回 None
    """
    if lon < -180 or lon > 180 or lat < -90 or lat > 90:
        return None
    zone = int(math.floor((lon + 180) / 6)) + 1
    if lat >= 0:
        return 32600 + zone
    else:
        return 32700 + zone
def wgs84_to_utm(lonlatalt, epsg):
    """
    WGS84 -> UTM
    参数
      lonlatalt: 形状为 (3,) 或 (N,3) 的 [lon, lat, alt] (单位: 度, 度, 米)
      epsg: 目标 UTM 坐标系 EPSG 号 (如 32633/32733 等)
    返回
      若输入 (3,): 返回 (x, y, z) 元组
      若输入 (N,3): 返回 (N,3) 数组，列为 [x, y, z]
    说明
      z 直接沿用 alt，不做大地高/正常高转换。
    """
    arr = np.asarray(lonlatalt, dtype=float)
    if arr.ndim == 1:  # (3,)
        arr = arr[None, :]  # -> (1,3)

    assert arr.shape[1] == 3, "lonlatalt must be (3,) or (N,3)"
    lon = arr[:, 0]
    lat = arr[:, 1]
    alt = arr[:, 2]

    crs_wgs84 = CRS.from_epsg(4326)
    crs_utm   = CRS.from_epsg(int(epsg))
    transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)

    x, y = transformer.transform(lon, lat)  # 支持向量化
    z = alt.copy()                          # 高程直接传递

    out = np.stack([x, y, z], axis=1)
    # 单点输入则返回元组，批量返回 (N,3) 数组
    return (out[0, 0], out[0, 1], out[0, 2]) if out.shape[0] == 1 else out
# def wgs84_to_utm(lon, lat, alt, epsg):
#     """
#     将 WGS84 坐标 (lon, lat, alt) 转换到对应 UTM 坐标系 (x, y, z)。
#     这里 alt 保持不变（不做大地高与正高转换）。
#     """
#     crs_wgs84 = CRS.from_epsg(4326)
#     crs_utm   = CRS.from_epsg(epsg)
#     transformer = Transformer.from_crs(crs_wgs84, crs_utm, always_xy=True)
#     x, y = transformer.transform(lon, lat)
#     return [x, y, alt]
if __name__=="__main__":
    target_point = [[545,953]]
    target_point = [[540,946]]
    # -122.009837 37.33356199999999 800.0 0.0 10.0 348.6897984108729
    # -73.988756 40.74804400000001 700.0 0.0  343.30037061085557
    
    # query_camera = [1920, 1080, 19.2, 10.8, 23.176]
    query_camera = [960, 540, 9.6, 5.4, 11.588]
    # translation = [-73.988756, 40.74804400000001, 700.0]
    # euler_angles = [40.0 ,0.0, 343.30037061085557]
    euler_angles, translation,_ = get_init("/media/ubuntu/PS20001/poses/switzerland_seq4@8@sunny@500.txt", start=126)
    render_pose = [translation, euler_angles]
    rgb_path = "/media/ubuntu/PS20001/images/switzerland_seq4@8@sunny@500/126_0.png"
    depth_pth = "/mnt/sda/MapScape/query/depth/switzerland_seq4@8@night@intensity3@500/126_1.png"
    depth_mat = cv2.imread(depth_pth, cv2.IMREAD_UNCHANGED)
    depth_mat = cv2.flip(depth_mat, 0)
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.resize(rgb, (depth_mat.shape[1], depth_mat.shape[0]))
    
    # depth_mat, scale = read_pfm(depth_pth)
    target_point = torch.tensor(target_point)
    render_K, _, render_height_px = get_intrinsic(query_camera) # 将depth图resize到查询图尺寸，因此内参一致
    
    render_K = np.float64(render_K)
    
    K_c2w =  np.linalg.inv(render_K)
    # Get render pose
    render_T = np.float64(get_pose_mat(render_pose))  #! 旋转矩阵转换到ECEF坐标系下
    render_T[:3, 1] = -render_T[:3, 1] 
    render_T[:3, 2] = -render_T[:3, 2] 
    
    pts_ecef, cols, _ = backproject_depth_to_ecef(
    depth_m=depth_mat,
    K_c2w=K_c2w,
    render_T=render_T,
    stride=40,
    d_min=0.1,
    d_max=64000.0,
    rgb=rgb
    )
    pts_wgs84 = ECEF_to_WGS84(pts_ecef)
    utm_id = get_utm_epsg_from_lonlat(pts_wgs84[0,0], pts_wgs84[0,1])
    pts_utm = wgs84_to_utm(pts_wgs84, utm_id)

    print("[info] points:", pts_ecef.shape)
    save_path = ""
    # ---- 可视化（快速散点）----
    # quick_scatter3d(pts_ecef, cols, max_points=200000)
    quick_scatter3d_minimal(pts_utm, cols, max_points=200000,
                        save_path="/home/ubuntu/Documents/1-pixel/dense_cloud_minimal.png")
    # ---- 保存PLY（可选）----
    # save_ply("dense_cloud_stride4.ply", pts_ecef, cols)


