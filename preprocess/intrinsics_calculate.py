import numpy as np

def compute_K_from_sensor(
    sensor_width_mm: float,
    sensor_height_mm: float,
    image_width_px: int,
    image_height_px: int,
    focal_length_mm: float,
    cx: float = None,
    cy: float = None
):
    """
    根据传感器尺寸 + 图像分辨率 + 物理焦距计算相机内参矩阵 K

    Parameters
    ----------
    sensor_width_mm : float
        传感器物理宽度 (mm)
    sensor_height_mm : float
        传感器物理高度 (mm)
    image_width_px : int
        图像宽度 (px)
    image_height_px : int
        图像高度 (px)
    focal_length_mm : float
        镜头物理焦距 (mm)
    cx, cy : float, optional
        主点坐标 (px)，若为 None，默认取图像中心

    Returns
    -------
    K : np.ndarray (3×3)
        相机内参矩阵
    """

    # 像素焦距
    fx = focal_length_mm * image_width_px / sensor_width_mm
    fy = fx
    # fy = focal_length_mm * image_height_px / sensor_height_mm

    # 主点
    if cx is None:
        cx = image_width_px / 2.0
    if cy is None:
        cy = image_height_px / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return K
sensor_w = 6.17    # mm
sensor_h = 4.55    # mm
img_w = 3840
img_h = 2160
f_mm = 4.5         # 镜头物理焦距

K = compute_K_from_sensor(sensor_w, sensor_h, img_w, img_h, f_mm)
print(K)
