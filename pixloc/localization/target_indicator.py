import logging
import os
import cv2
import pycolmap
from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
from pixloc.utils.transform import get_rotation_enu_in_ecef, WGS84_to_ECEF
from pixloc.utils.transform import ECEF_to_WGS84
from scipy.spatial.transform import Rotation as R
import copy
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

class QueryLocalizer:
    def __init__(self):
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
            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y

            cx = image_width_px / 2
            cy = image_height_px / 2
        elif len(camera) == 7:
            image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, f_mm = camera
            focal_ratio_x = f_mm / sensor_width_mm
            focal_ratio_y = f_mm / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y
            
        elif len(camera) == 8:
            image_width_px, image_height_px, cx, cy, sensor_width_mm, sensor_height_mm, fmm_x, fmm_y = camera
            focal_ratio_x = fmm_x / sensor_width_mm
            focal_ratio_y = fmm_y / sensor_height_mm
            
            fx = image_width_px * focal_ratio_x
            fy = image_height_px * focal_ratio_y

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
        fx, fy = fmm, fmm
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
        lon, lat, _ = render_pose[0]

        euler_angles = render_pose[1].copy()
        rot_ned_in_ecef = get_rotation_enu_in_ecef(lon, lat)
        rot_pose_in_ned = R.from_euler('xyz', euler_angles, degrees=True).as_matrix()
        
        r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
        
        xyz = WGS84_to_ECEF(render_pose[0])
        
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
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
            points_2D = points_2D.T
        else:
            points_2D = points.T

        t = t.unsqueeze(1)
        t = t.repeat(1, points_2D.size(-1))

        points_2D = points_2D.float()
        K = K.float()
        R = R.float()
        depth = depth.float()
        t = t.float()

        K[-1, -1] = -1

        Points_3D = R @ (K @ (depth * points_2D)) + t
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
        if points.shape[-1] != 3:
            points_2D = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1)
            points_2D = points_2D.T
        else:
            points_2D = points.T

        t = t.unsqueeze(1)
        t = t.repeat(1, points_2D.size(-1))

        points_2D = points_2D.float()
        K = K.float()
        R = R.float()
        depth = depth.float()
        t = t.float()

        Points_3D = R @ (K @ (depth * points_2D)) + t
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
        if points.shape[-1] != 3:
            points_2D = np.concatenate([points, np.ones_like(points[ :, [0]])], axis=-1)
            points_2D = points_2D.T
        else:
            points_2D = points.T

        
        t = np.expand_dims(t,-1)
        t = np.tile(t, points_2D.shape[-1])

        points_2D = np.float64(points_2D)
        K = np.float64(K)
        R = np.float64(R)
        depth = np.float64(depth)
        t = np.float64(t)

        K[-1, -1] = -1
        
        Points_3D = R @ K @ (depth * points_2D) + t
        return Points_3D.T
    def get_new_focal_lens(self, R, t, K, points_3D, points):
        x, y = points
        points_3D = np.float64(points_3D)
        K = np.float64(K)
        R = np.float64(R)
        t = np.float64(t)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        R_inverse = np.linalg.inv(R)
        point_3d_camera = np.expand_dims(points_3D - t, 1)
        point_3d_camera_r = R_inverse @ point_3d_camera
        Rx, Ry, Rz = point_3d_camera_r
        
        fx_new = (x - cx) * Rz / Rx
        fy_new = (y - cy) * Rz / Ry
        logger.debug("Rx=%.4f, Ry=%.4f, Rz=%.4f", Rx, Ry, Rz)
        logger.debug("fx_new=%.4f, fy_new=%.4f", fx_new, fy_new)
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
        points_3D = torch.tensor(points_3D, dtype=torch.float64)
        K = torch.tensor(K, dtype=torch.float64)
        R = torch.tensor(R, dtype=torch.float64)
        t = torch.tensor(t, dtype=torch.float64)
        t = t.unsqueeze(1)
        t = t.repeat(1, points_3D.size(0)).transpose(0, 1)

        K[-1, -1] = -1

        point_3d_camera = points_3D - t
        point_3d_camera_r = torch.matmul(R.inverse(), point_3d_camera.transpose(0, 1))
        point_2d_homo = torch.matmul(K.inverse(), torch.cat((point_3d_camera_r, torch.ones(points_3D.shape[0], 1, device=points_3D.device)), dim=1))
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
        points_3D = np.float64(points_3D)
        K = np.float64(K)
        R = np.float64(R)
        t = np.float64(t)
        K[-1, -1] = -1
        
        K_inverse = np.linalg.inv(K)
        R_inverse = np.linalg.inv(R)
        point_3d_camera = np.expand_dims(points_3D - t, 1)
        point_3d_camera_r = R_inverse @ point_3d_camera
        point_2d_homo = K_inverse @ point_3d_camera_r
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
        points_3D = np.float64(points_3D)
        K = np.float64(K)
        R = np.float64(R)
        t = np.float64(t)
        # K[-1, -1] = -1
        
        K_inverse = np.linalg.inv(K)
        R_inverse = np.linalg.inv(R)
        point_3d_camera = np.expand_dims(points_3D - t, 1)
        point_3d_camera_r = R_inverse @ point_3d_camera
        point_2d_homo = K_inverse @ point_3d_camera_r
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
        ret['qvec'] = rot_mat.transpose()
        ret['tvec'] = np.array(-t_c2w).reshape(3)
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
        
    def get_target_location(self, target_point, render_pose, depth_mat, camera):
        target_point = torch.tensor(target_point)

        fx, fy = float(camera.f[0]), float(camera.f[1])
        cx, cy = float(camera.c[0]), float(camera.c[1])

        render_K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1]
        ], dtype=np.float64)
        
        K_c2w = np.linalg.inv(render_K)
        # Get render pose
        render_T = np.float64(self.get_pose_mat(render_pose))
        
        depth, _ = self.read_valid_depth(target_point, depth = depth_mat)
        
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
    
  
    
def vis_points(points_3d):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    x_coords = [point[0] for point in points_3d]
    y_coords = [point[1] for point in points_3d]
    z_coords = [point[2] for point in points_3d]

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x_coords, y_coords, z_coords)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
def vis_map(point3d, point2d):
    import matplotlib.pyplot as plt
    import mplcursors

    image_path = '/home/ubuntu/Documents/code/github/target_indicator_server/fast_render2loc/datasets/松兰山/prior_images/Z_0_42.png'
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots()
    img_plot = ax.imshow(image_rgb) 
    points = {}

    for i in range(len(point3d)):
        x = int(point2d[i][0])
        y = int(point2d[i][1])
        info = str(point3d[i])
        points[(x, y)] = info
    def on_hover(sel):
        y, x = sel.target.index
        logger.debug("xy: (%s, %s)", x, y)
        if (x, y) in points:
            sel.annotation.set_text(points[(x, y)])
        else:
            sel.annotation.set_text('No info')
    cursor = mplcursors.cursor(hover=True)
    cursor.connect("add", on_hover)

    plt.show()
    

# if __name__=="__main__":
#     main()


