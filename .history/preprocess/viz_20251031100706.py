import trimesh
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import PIL.Image
OPENGL = np.array([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

def uint8(colors):
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors *= 255
    assert 0 <= colors.min() and colors.max() < 256
    return np.uint8(colors)

def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def add_scene_cam(scene, pose_c2w, edge_color, image=None, focal=None, imsize=None, 
                  screen_width=0.03, marker=None):
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255*image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H,W) * 1.1 # default value

    # create fake camera
    height = max( screen_width/10, focal * screen_width / H )
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W/H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    # this is the image
    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
        scene.add_geometry(img)

    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    THICKNESS_SCALE = 0.95  # 原始值是 0.95
    vertices = np.r_[cam.vertices, THICKNESS_SCALE * cam.vertices, geotrf(rot2, cam.vertices)]
    # vertices = np.r_[cam.vertices, 0.95*cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]
    BRIGHT_YELLOW = (255, 255, 0)
    GREEN = (0, 255, 0)

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = GREEN
    scene.add_geometry(cam)
    if marker == 'o':
        marker = trimesh.creation.icosphere(3, radius=screen_width/4)
        marker.vertices += pose_c2w[:3,3]
        marker.visual.face_colors[:,:3] = GREEN
        scene.add_geometry(marker)


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask

def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask

def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x): return todevice(x, 'numpy')

def img_to_arr(img):
    if isinstance(img, str):
        img = imread_cv2(img)
    return img


def imread_cv2(path, options=cv2.IMREAD_COLOR):
    """ Open an image or a depthmap with opencv-python.
    """
    if path.endswith(('.exr', 'EXR')):
        options = cv2.IMREAD_ANYDEPTH
    img = cv2.imread(path, options)
    if img is None:
        raise IOError(f'Could not load image={path} with {options=}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_poses(pose_file):
    """Load poses from the pose file."""
    pose_dict = {}
    translation_list = []
    euler_angles_list = []
    name_list = []
    with open(pose_file, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line
            parts = line.strip().split()
            if parts:  # Ensure the line is not empty
                # pose_dict[parts[0]] = parts[1: ]  # Add the first element to the name list
                lon, lat, alt, roll, pitch, yaw = map(float, parts[1: ])
                translation_list.append([lon, lat, alt])
                euler_angles_list.append([pitch, roll, yaw])
                name_list.append(parts[0].split('.')[0]+'_0.png')

    # Convert to numpy arrays
    translation_array = np.array(translation_list)  # shape (n, 3)
    euler_angles_array = np.array(euler_angles_list)  # shape (n, 3)
    T_c2w = get_matrix_list_batch(translation_array, euler_angles_array)
    for i in range(len(name_list)):
        pose_dict[name_list[i]] = {}
        pose_dict[name_list[i]]['euler_angles'] = euler_angles_array[i].tolist()
        pose_dict[name_list[i]]['translation'] = translation_array[i].tolist()
        pose_dict[name_list[i]]['T_c2w'] = T_c2w[i]
    return pose_dict
class SceneViz:
    def __init__(self):
        self.scene = trimesh.Scene()

    def add_rgbd(self, image, depth, intrinsics=None, cam2world=None, zfar=np.inf, mask=None):
        image = img_to_arr(image)

        # make up some intrinsics
        if intrinsics is None:
            H, W, THREE = image.shape
            focal = max(H, W)
            intrinsics = np.float32([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])

        # compute 3d points
        pts3d = depthmap_to_pts3d(depth, intrinsics, cam2world=cam2world)

        return self.add_pointcloud(pts3d, image, mask=(depth<zfar) if mask is None else mask)

    def add_pointcloud(self, pts3d, color=(0,0,0), mask=None, denoise=False):
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)
        if not isinstance(pts3d, list):
            pts3d = [pts3d.reshape(-1,3)]
            if mask is not None: 
                mask = [mask.ravel()]
        if not isinstance(color, (tuple,list)):
            color = [color.reshape(-1,3)]
        if mask is None:
            mask = [slice(None)] * len(pts3d)

        pts = np.concatenate([p[m] for p,m in zip(pts3d,mask)])
        pct = trimesh.PointCloud(pts)

        if isinstance(color, (list, np.ndarray, torch.Tensor)):
            color = to_numpy(color)
            col = np.concatenate([p[m] for p,m in zip(color,mask)])
            assert col.shape == pts.shape, bb()
            pct.visual.vertex_colors = uint8(col.reshape(-1,3))
        else:
            assert len(color) == 3
            pct.visual.vertex_colors = np.broadcast_to(uint8(color), pts.shape)

        if denoise:
            # remove points which are noisy
            centroid = np.median(pct.vertices, axis=0)
            dist_to_centroid = np.linalg.norm( pct.vertices - centroid, axis=-1)
            dist_thr = np.quantile(dist_to_centroid, 0.99)
            valid = (dist_to_centroid < dist_thr)
            # new cleaned pointcloud
            pct = trimesh.PointCloud(pct.vertices[valid], color=pct.visual.vertex_colors[valid])

        self.scene.add_geometry(pct)
        return self

    def add_rgbd(self, image, depth, intrinsics=None, cam2world=None, zfar=np.inf, mask=None):
        # make up some intrinsics
        if intrinsics is None:
            H, W, THREE = image.shape
            focal = max(H, W)
            intrinsics = np.float32([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])

        # compute 3d points
        pts3d, mask2 = depthmap_to_absolute_camera_coordinates(depth, intrinsics, cam2world)
        mask2 &= (depth<zfar) 

        # combine with provided mask if any
        if mask is not None:
            mask2 &= mask

        return self.add_pointcloud(pts3d, image, mask=mask2)

    def add_camera(self, pose_c2w, focal=None, color=(0, 0, 0), image=None, imsize=None, cam_size=0.03):
        pose_c2w, focal, color, image = to_numpy((pose_c2w, focal, color, image))
        image = img_to_arr(image)
        if isinstance(focal, np.ndarray) and focal.shape == (3,3):
            intrinsics = focal
            focal = (intrinsics[0,0] * intrinsics[1,1]) ** 0.5
            if imsize is None:
                imsize = (2*intrinsics[0,2], 2*intrinsics[1,2])
        
        add_scene_cam(self.scene, pose_c2w, color, image, focal, imsize=imsize, screen_width=cam_size, marker=None)
        return self

    def add_cameras(self, poses, focals=None, images=None, imsizes=None, colors=None, **kw):
        get = lambda arr,idx: None if arr is None else arr[idx]
        for i, pose_c2w in enumerate(poses):
            self.add_camera(pose_c2w, get(focals,i), image=get(images,i), color=get(colors,i), imsize=get(imsizes,i), **kw)
        return self

    def show(self, point_size=2):
        self.scene.show(line_settings= {'point_size': point_size})