from typing import *
from numbers import Number 

import torch
import torch.nn.functional as F

from ._helpers import batched


__all__ = [
    'perspective',
    'perspective_from_fov',
    'perspective_from_fov_xy',
    'intrinsics_from_focal_center',
    'intrinsics_from_fov',
    'intrinsics_from_fov_xy',
    'focal_to_fov',
    'fov_to_focal',
    'intrinsics_to_fov',
    'view_look_at',
    'extrinsics_look_at',
    'perspective_to_intrinsics',
    'intrinsics_to_perspective',
    'extrinsics_to_view',
    'view_to_extrinsics',
    'normalize_intrinsics',
    'crop_intrinsics',
    'pixel_to_uv',
    'pixel_to_ndc',
    'uv_to_pixel',
    'project_depth',
    'depth_buffer_to_linear',
    'project_gl',
    'project_cv',
    'unproject_gl',
    'unproject_cv',
    'skew_symmetric',
    'rotation_matrix_from_vectors',
    'euler_axis_angle_rotation',
    'euler_angles_to_matrix',
    'matrix_to_euler_angles',
    'matrix_to_quaternion',
    'quaternion_to_matrix',
    'matrix_to_axis_angle',
    'axis_angle_to_matrix',
    'axis_angle_to_quaternion',
    'quaternion_to_axis_angle',
    'slerp',
    'interpolate_extrinsics',
    'interpolate_view',
    'extrinsics_to_essential',
    'to4x4',
    'rotation_matrix_2d',
    'rotate_2d',
    'translate_2d',
    'scale_2d',
    'apply_2d',
]


def closed_form_inverse_intrinsics(intrinsics):
    """
    Compute the inverse of each 3x3 intrinsics matrix in a batch.
    """
    inv_intrinsics = torch.zeros_like(intrinsics)
    inv_intrinsics[..., 0, 0] = 1. / intrinsics[..., 0, 0]
    inv_intrinsics[..., 1, 1] = 1. / intrinsics[..., 1, 1]
    inv_intrinsics[..., 0, 2] = - intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    inv_intrinsics[..., 1, 2] = - intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    inv_intrinsics[..., 2, 2] = 1.
    return inv_intrinsics


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 tensor of SE3 matrices.
        R (optional): Nx3x3 tensor of rotation matrices.
        T (optional): Nx3x1 tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4) or (..., 4, 4)
        R: (N, 3, 3) or (..., 3, 3)
        T: (N, 3, 1) or (..., 3, 1)
    """
    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    ori_shape = se3.shape

    # Extract R and T if not provided
    if R is None:
        R = se3[..., :3, :3].reshape(-1, 3, 3)
    if T is None:
        T = se3[..., :3, 3:].reshape(-1, 3, 1)

    # Transpose R
    R_transposed = R.transpose(1, 2)  # (N,3,3)
    top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
    inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
    inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[..., :3, :3] = R_transposed
    inverted_matrix[..., :3, 3:] = top_right

    return inverted_matrix.reshape(ori_shape)


@batched(0,0,0,0)
def perspective(
        fov_y: Union[float, torch.Tensor],
        aspect: Union[float, torch.Tensor],
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenGL perspective matrix

    Args:
        fov_y (float | torch.Tensor): field of view in y axis
        aspect (float | torch.Tensor): aspect ratio
        near (float | torch.Tensor): near plane to clip
        far (float | torch.Tensor): far plane to clip

    Returns:
        (torch.Tensor): [..., 4, 4] perspective matrix
    """
    N = fov_y.shape[0]
    ret = torch.zeros((N, 4, 4), dtype=fov_y.dtype, device=fov_y.device)
    ret[:, 0, 0] = 1. / (torch.tan(fov_y / 2) * aspect)
    ret[:, 1, 1] = 1. / (torch.tan(fov_y / 2))
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret


def perspective_from_fov(
        fov: Union[float, torch.Tensor],
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor],
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenGL perspective matrix from field of view in largest dimension

    Args:
        fov (float | torch.Tensor): field of view in largest dimension
        width (int | torch.Tensor): image width
        height (int | torch.Tensor): image height
        near (float | torch.Tensor): near plane to clip
        far (float | torch.Tensor): far plane to clip

    Returns:
        (torch.Tensor): [..., 4, 4] perspective matrix
    """
    fov_y = 2 * torch.atan(torch.tan(fov / 2) * height / torch.maximum(width, height))
    aspect = width / height
    return perspective(fov_y, aspect, near, far)


def perspective_from_fov_xy(
        fov_x: Union[float, torch.Tensor],
        fov_y: Union[float, torch.Tensor],
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenGL perspective matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis
        near (float | torch.Tensor): near plane to clip
        far (float | torch.Tensor): far plane to clip

    Returns:
        (torch.Tensor): [..., 4, 4] perspective matrix
    """
    aspect = torch.tan(fov_x / 2) / torch.tan(fov_y / 2)
    return perspective(fov_y, aspect, near, far)


@batched(0,0,0,0)
def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    Args:
        focal_x (float | torch.Tensor): focal length in x axis
        focal_y (float | torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    N = fx.shape[0]
    ret = torch.zeros((N, 3, 3), dtype=fx.dtype, device=fx.device)
    zeros, ones = torch.zeros(N, dtype=fx.dtype, device=fx.device), torch.ones(N, dtype=fx.dtype, device=fx.device)
    ret = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))
    return ret


@batched(0, 0, 0, 0, 0, 0)
def intrinsics_from_fov(
    fov_max: Union[float, torch.Tensor] = None,
    fov_min: Union[float, torch.Tensor] = None,
    fov_x: Union[float, torch.Tensor] = None,
    fov_y: Union[float, torch.Tensor] = None,
    width: Union[int, torch.Tensor] = None,
    height: Union[int, torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get normalized OpenCV intrinsics matrix from given field of view.
    You can provide either fov_max, fov_min, fov_x or fov_y

    Args:
        width (int | torch.Tensor): image width
        height (int | torch.Tensor): image height
        fov_max (float | torch.Tensor): field of view in largest dimension
        fov_min (float | torch.Tensor): field of view in smallest dimension
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    if fov_max is not None:
        fx = torch.maximum(width, height) / width / (2 * torch.tan(fov_max / 2))
        fy = torch.maximum(width, height) / height / (2 * torch.tan(fov_max / 2))
    elif fov_min is not None:
        fx = torch.minimum(width, height) / width / (2 * torch.tan(fov_min / 2))
        fy = torch.minimum(width, height) / height / (2 * torch.tan(fov_min / 2))
    elif fov_x is not None and fov_y is not None:
        fx = 1 / (2 * torch.tan(fov_x / 2))
        fy = 1 / (2 * torch.tan(fov_y / 2))
    elif fov_x is not None:
        fx = 1 / (2 * torch.tan(fov_x / 2))
        fy = fx * width / height
    elif fov_y is not None:
        fy = 1 / (2 * torch.tan(fov_y / 2))
        fx = fy * height / width
    cx = 0.5
    cy = 0.5
    ret = intrinsics_from_focal_center(fx, fy, cx, cy)
    return ret



def intrinsics_from_fov_xy(
    fov_x: Union[float, torch.Tensor],
    fov_y: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    cx = cy = 0.5
    return intrinsics_from_focal_center(focal_x, focal_y, cx, cy)


def focal_to_fov(focal: torch.Tensor):
    return 2 * torch.atan(0.5 / focal)


def fov_to_focal(fov: torch.Tensor):
    return 0.5 / torch.tan(fov / 2)


def intrinsics_to_fov(intrinsics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    "NOTE: approximate FOV by assuming centered principal point"
    fov_x = focal_to_fov(intrinsics[..., 0, 0])
    fov_y = focal_to_fov(intrinsics[..., 1, 1])
    return fov_x, fov_y


@batched(1,1,1)
def view_look_at(
    eye: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """
    Get OpenGL view matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], view matrix
    """
    N = eye.shape[0]
    z = eye - look_at
    x = torch.cross(up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.
    return ret


@batched(1, 1, 1)
def extrinsics_look_at(
    eye: torch.Tensor,
    look_at: torch.Tensor,
    up: torch.Tensor
) -> torch.Tensor:
    """
    Get OpenCV extrinsics matrix looking at something

    Args:
        eye (torch.Tensor): [..., 3] the eye position
        look_at (torch.Tensor): [..., 3] the position to look at
        up (torch.Tensor): [..., 3] head up direction (-y axis in screen space). Not necessarily othogonal to view direction

    Returns:
        (torch.Tensor): [..., 4, 4], extrinsics matrix
    """
    N = eye.shape[0]
    z = look_at - eye
    x = torch.cross(-up, z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    # x = torch.cross(y, z, dim=-1)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    z = z / z.norm(dim=-1, keepdim=True)
    R = torch.stack([x, y, z], dim=-2)
    t = -torch.matmul(R, eye[..., None])
    ret = torch.zeros((N, 4, 4), dtype=eye.dtype, device=eye.device)
    ret[:, :3, :3] = R
    ret[:, :3, 3] = t[:, :, 0]
    ret[:, 3, 3] = 1.
    return ret


@batched(2)
def perspective_to_intrinsics(
    perspective: torch.Tensor
) -> torch.Tensor:
    """
    OpenGL perspective matrix to OpenCV intrinsics

    Args:
        perspective (torch.Tensor): [..., 4, 4] OpenGL perspective matrix

    Returns:
        (torch.Tensor): shape [..., 3, 3] OpenCV intrinsics
    """
    assert torch.allclose(perspective[:, [0, 1, 3], 3], 0), "The perspective matrix is not a projection matrix"
    ret = torch.tensor([[0.5, 0., 0.5], [0., -0.5, 0.5], [0., 0., 1.]], dtype=perspective.dtype, device=perspective.device) \
        @ perspective[:, [0, 1, 3], :3] \
        @ torch.diag(torch.tensor([1, -1, -1], dtype=perspective.dtype, device=perspective.device))
    return ret / ret[:, 2, 2, None, None]


@batched(2,0,0)
def intrinsics_to_perspective(
    intrinsics: torch.Tensor,
    near: Union[float, torch.Tensor],
    far: Union[float, torch.Tensor],
) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip
    Returns:
        (torch.Tensor): [..., 4, 4] OpenGL perspective matrix
    """
    N = intrinsics.shape[0]
    fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
    cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    ret = torch.zeros((N, 4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[:, 0, 0] = 2 * fx
    ret[:, 1, 1] = 2 * fy
    ret[:, 0, 2] = -2 * cx + 1
    ret[:, 1, 2] = 2 * cy - 1
    ret[:, 2, 2] = (near + far) / (near - far)
    ret[:, 2, 3] = 2. * near * far / (near - far)
    ret[:, 3, 2] = -1.
    return ret


@batched(2)
def extrinsics_to_view(
        extrinsics: torch.Tensor
    ) -> torch.Tensor:
    """
    OpenCV camera extrinsics to OpenGL view matrix

    Args:
        extrinsics (torch.Tensor): [..., 4, 4] OpenCV camera extrinsics matrix

    Returns:
        (torch.Tensor): [..., 4, 4] OpenGL view matrix
    """
    return extrinsics * torch.tensor([1, -1, -1, 1], dtype=extrinsics.dtype, device=extrinsics.device)[:, None]


@batched(2)
def view_to_extrinsics(
        view: torch.Tensor
    ) -> torch.Tensor:
    """
    OpenGL view matrix to OpenCV camera extrinsics

    Args:
        view (torch.Tensor): [..., 4, 4] OpenGL view matrix

    Returns:
        (torch.Tensor): [..., 4, 4] OpenCV camera extrinsics matrix
    """
    return view  * torch.tensor([1, -1, -1, 1], dtype=view.dtype, device=view.device)[:, None]


@batched(2,0,0)
def normalize_intrinsics(
        intrinsics: torch.Tensor,
        width: Union[int, torch.Tensor],
        height: Union[int, torch.Tensor]
    ) -> torch.Tensor:
    """
    Normalize camera intrinsics(s) to uv space

    Args:
        intrinsics (torch.Tensor): [..., 3, 3] camera intrinsics(s) to normalize
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)

    Returns:
        (torch.Tensor): [..., 3, 3] normalized camera intrinsics(s)
    """
    zeros = torch.zeros_like(width)
    ones = torch.ones_like(width)
    transform = torch.stack([
        1 / width, zeros, 0.5 / width,
        zeros, 1 / height, 0.5 / height,
        zeros, zeros, ones
    ]).reshape(*zeros.shape, 3, 3).to(intrinsics)
    return transform @ intrinsics



@batched(2,0,0,0,0,0,0)
def crop_intrinsics(
    intrinsics: torch.Tensor,
    width: Union[int, torch.Tensor],
    height: Union[int, torch.Tensor],
    left: Union[int, torch.Tensor],
    top: Union[int, torch.Tensor],
    crop_width: Union[int, torch.Tensor],
    crop_height: Union[int, torch.Tensor]
) -> torch.Tensor:
    """
    Evaluate the new intrinsics(s) after crop the image: cropped_img = img[top:top+crop_height, left:left+crop_width]

    Args:
        intrinsics (torch.Tensor): [..., 3, 3] camera intrinsics(s) to crop
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)
        left (int | torch.Tensor): [...] left crop boundary
        top (int | torch.Tensor): [...] top crop boundary
        crop_width (int | torch.Tensor): [...] crop width
        crop_height (int | torch.Tensor): [...] crop height

    Returns:
        (torch.Tensor): [..., 3, 3] cropped camera intrinsics(s)
    """
    zeros = torch.zeros_like(width)
    ones = torch.ones_like(width)
    transform = torch.stack([
        width / crop_width, zeros, -left / crop_width,
        zeros, height / crop_height, -top / crop_height,
        zeros, zeros, ones
    ]).reshape(*zeros.shape, 3, 3).to(intrinsics)
    return transform @ intrinsics


@batched(1,0,0)
def pixel_to_uv(
    pixel: torch.Tensor,
    width: Union[int, torch.Tensor],
    height: Union[int, torch.Tensor]
) -> torch.Tensor:
    """
    Args:
        pixel (torch.Tensor): [..., 2] pixel coordinrates defined in image space,  x range is (0, W - 1), y range is (0, H - 1)
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)

    Returns:
        (torch.Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    if not torch.is_floating_point(pixel):
        pixel = pixel.float()
    uv = (pixel + 0.5) / torch.stack([width, height], dim=-1).to(pixel)
    return uv


@batched(1,0,0)
def uv_to_pixel(
    uv: torch.Tensor,
    width: Union[int, torch.Tensor],
    height: Union[int, torch.Tensor]
) -> torch.Tensor:
    """
    Args:
        uv (torch.Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)

    Returns:
        (torch.Tensor): [..., 2] pixel coordinrates defined in uv space, the range is (0, 1)
    """
    pixel = uv * torch.stack([width, height], dim=-1).to(uv) - 0.5
    return pixel


@batched(1,0,0)
def pixel_to_ndc(
    pixel: torch.Tensor,
    width: Union[int, torch.Tensor],
    height: Union[int, torch.Tensor]
) -> torch.Tensor:
    """
    Args:
        pixel (torch.Tensor): [..., 2] pixel coordinrates defined in image space, x range is (0, W - 1), y range is (0, H - 1)
        width (int | torch.Tensor): [...] image width(s)
        height (int | torch.Tensor): [...] image height(s)

    Returns:
        (torch.Tensor): [..., 2] pixel coordinrates defined in ndc space, the range is (-1, 1)
    """
    if not torch.is_floating_point(pixel):
        pixel = pixel.float()
    ndc = (pixel + 0.5) / (torch.stack([width, height], dim=-1).to(pixel) * torch.tensor([2, -2], dtype=pixel.dtype, device=pixel.device)) \
        + torch.tensor([-1, 1], dtype=pixel.dtype, device=pixel.device)
    return ndc


@batched(0,0,0)
def project_depth(
        depth: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Project linear depth to depth value in screen space

    Args:
        depth (torch.Tensor): [...] depth value
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip

    Returns:
        (torch.Tensor): [..., 1] depth value in screen space, value ranging in [0, 1]
    """
    return (far - near * far / depth) / (far - near)


@batched(0,0,0)
def depth_buffer_to_linear(
        depth: torch.Tensor,
        near: Union[float, torch.Tensor],
        far: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Linearize depth value to linear depth

    Args:
        depth (torch.Tensor): [...] screen depth value, ranging in [0, 1]
        near (float | torch.Tensor): [...] near plane to clip
        far (float | torch.Tensor): [...] far plane to clip

    Returns:
        (torch.Tensor): [...] linear depth
    """
    return near * far / (far - (far - near) * depth)


@batched(2, 2, 2, 2)
def project_gl(
    points: torch.Tensor,
    model: torch.Tensor = None,
    view: torch.Tensor = None,
    perspective: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D following the OpenGL convention (except for row major matrice)

    Args:
        points (torch.Tensor): [..., N, 3 or 4] 3D points to project, if the last 
            dimension is 4, the points are assumed to be in homogeneous coordinates
        model (torch.Tensor): [..., 4, 4] model matrix
        view (torch.Tensor): [..., 4, 4] view matrix
        perspective (torch.Tensor): [..., 4, 4] perspective matrix

    Returns:
        scr_coord (torch.Tensor): [..., N, 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        linear_depth (torch.Tensor): [..., N] linear depth
    """
    assert perspective is not None, "perspective matrix is required"

    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    mvp = perspective if perspective is not None else torch.eye(4).to(points)
    if view is not None:
        mvp = mvp @ view
    if model is not None:
        mvp = mvp @ model
    clip_coord = points @ mvp.transpose(-1, -2)
    ndc_coord = clip_coord[..., :3] / clip_coord[..., 3:]
    scr_coord = ndc_coord * 0.5 + 0.5
    linear_depth = clip_coord[..., 3]
    return scr_coord, linear_depth


@batched(2, 2, 2)
def project_cv(
    points: torch.Tensor,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Project 3D points to 2D following the OpenCV convention

    Args:
        points (torch.Tensor): [..., N, 3] or [..., N, 4] 3D points to project, if the last
            dimension is 4, the points are assumed to be in homogeneous coordinates
        extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix
        intrinsics (torch.Tensor): [..., 3, 3] intrinsics matrix

    Returns:
        uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        linear_depth (torch.Tensor): [..., N] linear depth
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    if extrinsics is not None:
        points = points @ extrinsics.transpose(-1, -2)
    points = points[..., :3] @ intrinsics.transpose(-2, -1)
    uv_coord = points[..., :2] / points[..., 2:]
    linear_depth = points[..., 2]
    return uv_coord, linear_depth


@batched(2, 2, 2, 2)
def unproject_gl(
        screen_coord: torch.Tensor,
        model: torch.Tensor = None,
        view: torch.Tensor = None,
        perspective: torch.Tensor = None
    ) -> torch.Tensor:
    """
    Unproject screen space coordinates to 3D view space following the OpenGL convention (except for row major matrice)

    Args:
        screen_coord (torch.Tensor): [... N, 3] screen space coordinates, value ranging in [0, 1].
            The origin (0., 0., 0.) is corresponding to the left & bottom & nearest
        model (torch.Tensor): [..., 4, 4] model matrix
        view (torch.Tensor): [..., 4, 4] view matrix
        perspective (torch.Tensor): [..., 4, 4] perspective matrix

    Returns:
        points (torch.Tensor): [..., N, 3] 3d points
    """
    assert perspective is not None, "perspective matrix is required"
    ndc_xy = screen_coord * 2 - 1
    clip_coord = torch.cat([ndc_xy, torch.ones_like(ndc_xy[..., :1])], dim=-1)
    transform = perspective
    if view is not None:
        transform = transform @ view
    if model is not None:
        transform = transform @ model
    transform = torch.inverse(transform)
    points = clip_coord @ transform.transpose(-1, -2)
    points = points[..., :3] / points[..., 3:]
    return points
    

@batched(2, 1, 2, 2)
def unproject_cv(
    uv_coord: torch.Tensor,
    depth: torch.Tensor = None,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None
) -> torch.Tensor:
    """
    Unproject uv coordinates to 3D view space following the OpenCV convention

    Args:
        uv_coord (torch.Tensor): [..., N, 2] uv coordinates, value ranging in [0, 1].
            The origin (0., 0.) is corresponding to the left & top
        depth (torch.Tensor): [..., N] depth value
        extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix
        intrinsics (torch.Tensor): [..., 3, 3] intrinsics matrix

    Returns:
        points (torch.Tensor): [..., N, 3] 3d points
    """
    assert intrinsics is not None, "intrinsics matrix is required"
    points = torch.cat([uv_coord, torch.ones_like(uv_coord[..., :1])], dim=-1)
    # points = points @ torch.inverse(intrinsics).transpose(-2, -1)
    points = points @ closed_form_inverse_intrinsics(intrinsics).transpose(-2, -1)
    if depth is not None:
        points = points * depth[..., None]
    if extrinsics is not None:
        points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
        # points = (points @ torch.inverse(extrinsics).transpose(-2, -1))[..., :3]
        points = (points @ closed_form_inverse_se3(extrinsics).transpose(-2, -1))[..., :3]
    return points


def euler_axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str = 'XYZ') -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3), XYZ
        convention: permutation of "X", "Y" or "Z", representing the order of Euler rotations to apply.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        euler_axis_angle_rotation(c, euler_angles[..., 'XYZ'.index(c)])
        for c in convention
    ]
    # return functools.reduce(torch.matmul, matrices)
    return matrices[2] @ matrices[1] @ matrices[0]


def skew_symmetric(v: torch.Tensor):
    "Skew symmetric matrix from a 3D vector"
    assert v.shape[-1] == 3, "v must be 3D"
    x, y, z = v.unbind(dim=-1)
    zeros = torch.zeros_like(x)
    return torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros,
    ], dim=-1).reshape(*v.shape[:-1], 3, 3)


def rotation_matrix_from_vectors(v1: torch.Tensor, v2: torch.Tensor):
    "Rotation matrix that rotates v1 to v2"
    I = torch.eye(3).to(v1)
    v1 = F.normalize(v1, dim=-1)
    v2 = F.normalize(v2, dim=-1)
    v = torch.cross(v1, v2, dim=-1)
    c = torch.sum(v1 * v2, dim=-1)
    K = skew_symmetric(v)
    R = I + K + (1 / (1 + c))[None, None] * (K @ K)
    return R


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.
    NOTE: The composition order eg. `XYZ` means `Rz * Ry * Rx` (like blender), instead of `Rx * Ry * Rz` (like pytorch3d)

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3), in the order of XYZ (like blender), instead of convention (like pytorch3d)
    """
    if not all(c in 'XYZ' for c in convention) or not all(c in convention for c in 'XYZ'):
        raise ValueError(f"Invalid convention {convention}.")
    if not matrix.shape[-2:] == (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    
    i0 = 'XYZ'.index(convention[0])
    i2 = 'XYZ'.index(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(matrix[..., i2, i0] * (-1.0 if i2 - i0 in [-1, 2] else 1.0))
    else:
        central_angle = torch.acos(matrix[..., i2, i2])

    # Angles in composition order
    o = [
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2, :], True, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0], False, tait_bryan
        ),
    ]
    return torch.stack([o[convention.index(c)] for c in 'XYZ'], -1)


def axis_angle_to_matrix(axis_angle: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert axis-angle representation (rotation vector) to rotation matrix, whose direction is the axis of rotation and length is the angle of rotation

    Args:
        axis_angle (torch.Tensor): shape (..., 3), axis-angle vcetors

    Returns:
        torch.Tensor: shape (..., 3, 3) The rotation matrices for the given axis-angle parameters
    """
    batch_shape = axis_angle.shape[:-1]
    device, dtype = axis_angle.device, axis_angle.dtype

    angle = torch.norm(axis_angle + eps, dim=-1, keepdim=True)
    axis = axis_angle / angle

    cos = torch.cos(angle)[..., None, :]
    sin = torch.sin(angle)[..., None, :]

    rx, ry, rz = torch.split(axis, 3, dim=-1)
    zeros = torch.zeros((*batch_shape, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=-1).view((*batch_shape, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device)
    rot_mat = ident + sin * K + (1 - cos) * torch.matmul(K, K)
    return rot_mat


def matrix_to_axis_angle(rot_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert a batch of 3x3 rotation matrices to axis-angle representation (rotation vector)

    Args:
        rot_mat (torch.Tensor): shape (..., 3, 3), the rotation matrices to convert

    Returns:
        torch.Tensor: shape (..., 3), the axis-angle vectors corresponding to the given rotation matrices
    """
    quat = matrix_to_quaternion(rot_mat)
    axis_angle = quaternion_to_axis_angle(quat, eps=eps)
    return axis_angle


def quaternion_to_axis_angle(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert a batch of quaternions (w, x, y, z) to axis-angle representation (rotation vector)

    Args:
        quaternion (torch.Tensor): shape (..., 4), the quaternions to convert

    Returns:
        torch.Tensor: shape (..., 3), the axis-angle vectors corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    norm = torch.norm(quaternion[..., 1:], dim=-1, keepdim=True)
    axis = quaternion[..., 1:] / norm.clamp(min=eps)
    angle = 2 * torch.atan2(norm, quaternion[..., 0:1])
    return angle * axis


def axis_angle_to_quaternion(axis_angle: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert axis-angle representation (rotation vector) to quaternion (w, x, y, z)

    Args:
        axis_angle (torch.Tensor): shape (..., 3), axis-angle vcetors

    Returns:
        torch.Tensor: shape (..., 4) The quaternions for the given axis-angle parameters
    """
    axis = F.normalize(axis_angle, dim=-1, eps=eps)
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    quat = torch.cat([torch.cos(angle / 2), torch.sin(angle / 2) * axis], dim=-1)
    return quat


def matrix_to_quaternion(rot_mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)

    Args:
        rot_mat (torch.Tensor): shape (..., 3, 3), the rotation matrices to convert

    Returns:
        torch.Tensor: shape (..., 4), the quaternions corresponding to the given rotation matrices
    """
    # Extract the diagonal and off-diagonal elements of the rotation matrix
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = rot_mat.flatten(-2).unbind(dim=-1)

    diag = torch.diagonal(rot_mat, dim1=-2, dim2=-1)
    M = torch.tensor([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=rot_mat.dtype, device=rot_mat.device)
    wxyz = (1 + diag @ M.transpose(-1, -2)).clamp_(0).sqrt().mul(0.5)
    _, max_idx = wxyz.max(dim=-1)
    xw = torch.sign(m21 - m12)
    yw = torch.sign(m02 - m20)
    zw = torch.sign(m10 - m01)
    yz = torch.sign(m21 + m12)
    xz = torch.sign(m02 + m20)
    xy = torch.sign(m01 + m10)
    ones = torch.ones_like(xw)
    sign = torch.where(
        max_idx[..., None] == 0,
        torch.stack([ones, xw, yw, zw], dim=-1),
        torch.where(
            max_idx[..., None] == 1,
            torch.stack([xw, ones, xy, xz], dim=-1),
            torch.where(
                max_idx[..., None] == 2,
                torch.stack([yw, xy, ones, yz], dim=-1),
                torch.stack([zw, xz, yz, ones], dim=-1)
            )
        )
    )
    quat = sign * wxyz
    quat = F.normalize(quat, dim=-1, eps=eps)
    return quat


def quaternion_to_matrix(quaternion: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Converts a batch of quaternions (w, x, y, z) to rotation matrices
    
    Args:
        quaternion (torch.Tensor): shape (..., 4), the quaternions to convert
    
    Returns:
        torch.Tensor: shape (..., 3, 3), the rotation matrices corresponding to the given quaternions
    """
    assert quaternion.shape[-1] == 4
    quaternion = F.normalize(quaternion, dim=-1, eps=eps)
    w, x, y, z = quaternion.unbind(dim=-1)
    zeros = torch.zeros_like(w)
    I = torch.eye(3, dtype=quaternion.dtype, device=quaternion.device)
    xyz = quaternion[..., 1:]
    A = xyz[..., :, None] * xyz[..., None, :] - I * (xyz ** 2).sum(dim=-1)[..., None, None]
    B = torch.stack([
        zeros, -z, y,
        z, zeros, -x,
        -y, x, zeros
    ], dim=-1).unflatten(-1, (3, 3))
    rot_mat = I + 2 * (A + w[..., None, None] * B)
    return rot_mat


def slerp(rot_mat_1: torch.Tensor, rot_mat_2: torch.Tensor, t: Union[Number, torch.Tensor]) -> torch.Tensor:
    """Spherical linear interpolation between two rotation matrices

    Args:
        rot_mat_1 (torch.Tensor): shape (..., 3, 3), the first rotation matrix
        rot_mat_2 (torch.Tensor): shape (..., 3, 3), the second rotation matrix
        t (torch.Tensor): scalar or shape (...,), the interpolation factor

    Returns:
        torch.Tensor: shape (..., 3, 3), the interpolated rotation matrix
    """
    assert rot_mat_1.shape[-2:] == (3, 3)
    rot_vec_1 = matrix_to_axis_angle(rot_mat_1)
    rot_vec_2 = matrix_to_axis_angle(rot_mat_2)
    if isinstance(t, Number):
        t = torch.tensor(t, dtype=rot_mat_1.dtype, device=rot_mat_1.device)
    rot_vec = (1 - t[..., None]) * rot_vec_1 + t[..., None] * rot_vec_2
    rot_mat = axis_angle_to_matrix(rot_vec)
    return rot_mat


def interpolate_extrinsics(ext1: torch.Tensor, ext2: torch.Tensor, t: Union[Number, torch.Tensor]) -> torch.Tensor:
    """Interpolate extrinsics between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

    Args:
        ext1 (torch.Tensor): shape (..., 4, 4), the first camera pose
        ext2 (torch.Tensor): shape (..., 4, 4), the second camera pose
        t (torch.Tensor): scalar or shape (...,), the interpolation factor

    Returns:
        torch.Tensor: shape (..., 4, 4), the interpolated camera pose
    """
    return torch.inverse(interpolate_transform(torch.inverse(ext1), torch.inverse(ext2), t))


def interpolate_view(view1: torch.Tensor, view2: torch.Tensor, t: Union[Number, torch.Tensor]):
    """Interpolate view matrices between two camera poses. Linear interpolation for translation, spherical linear interpolation for rotation.

    Args:
        ext1 (torch.Tensor): shape (..., 4, 4), the first camera pose
        ext2 (torch.Tensor): shape (..., 4, 4), the second camera pose
        t (torch.Tensor): scalar or shape (...,), the interpolation factor

    Returns:
        torch.Tensor: shape (..., 4, 4), the interpolated camera pose
    """
    return interpolate_extrinsics(view1, view2, t)


def interpolate_transform(transform1: torch.Tensor, transform2: torch.Tensor, t: Union[Number, torch.Tensor]):
    assert transform1.shape[-2:] == (4, 4) and transform2.shape[-2:] == (4, 4)
    if isinstance(t, Number):
        t = torch.tensor(t, dtype=transform1.dtype, device=transform1.device)
    pos = (1 - t[..., None]) * transform1[..., :3, 3] + t[..., None] * transform2[..., :3, 3]
    rot = slerp(transform1[..., :3, :3], transform2[..., :3, :3], t)
    transform = torch.cat([rot, pos[..., None]], dim=-1)
    transform = torch.cat([ext, torch.tensor([0, 0, 0, 1], dtype=transform.dtype, device=transform.device).expand_as(transform[..., :1, :])], dim=-2)
    return transform


def extrinsics_to_essential(extrinsics: torch.Tensor):
    """
    extrinsics matrix `[[R, t] [0, 0, 0, 1]]` such that `x' = R (x - t)` to essential matrix such that `x' E x = 0`

    Args:
        extrinsics (torch.Tensor): [..., 4, 4] extrinsics matrix

    Returns:
        (torch.Tensor): [..., 3, 3] essential matrix
    """
    assert extrinsics.shape[-2:] == (4, 4)
    R = extrinsics[..., :3, :3]
    t = extrinsics[..., :3, 3]
    zeros = torch.zeros_like(t)
    t_x = torch.stack([
        zeros, -t[..., 2], t[..., 1],
        t[..., 2], zeros, -t[..., 0],
        -t[..., 1], t[..., 0], zeros
    ]).reshape(*t.shape[:-1], 3, 3)
    return R @ t_x


def to4x4(R: torch.Tensor, t: torch.Tensor):
    """
    Compose rotation matrix and translation vector to 4x4 transformation matrix

    Args:
        R (torch.Tensor): [..., 3, 3] rotation matrix
        t (torch.Tensor): [..., 3] translation vector

    Returns:
        (torch.Tensor): [..., 4, 4] transformation matrix
    """
    assert R.shape[-2:] == (3, 3)
    assert t.shape[-1] == 3
    assert R.shape[:-2] == t.shape[:-1]
    return torch.cat([
        torch.cat([R, t[..., None]], dim=-1),
        torch.tensor([0, 0, 0, 1], dtype=R.dtype, device=R.device).expand(*R.shape[:-2], 1, 4)
    ], dim=-2)


def rotation_matrix_2d(theta: Union[float, torch.Tensor]):
    """
    2x2 matrix for 2D rotation

    Args:
        theta (float | torch.Tensor): rotation angle in radians, arbitrary shape (...,)

    Returns:
        (torch.Tensor): (..., 2, 2) rotation matrix
    """
    if isinstance(theta, float):
        theta = torch.tensor(theta)
    return torch.stack([
        torch.cos(theta), -torch.sin(theta),
        torch.sin(theta), torch.cos(theta),
    ], dim=-1).unflatten(-1, (2, 2))


def rotate_2d(theta: Union[float, torch.Tensor], center: torch.Tensor = None):
    """
    3x3 matrix for 2D rotation around a center
    ```
       [[Rxx, Rxy, tx],
        [Ryx, Ryy, ty],
        [0,     0,  1]]
    ```
    Args:
        theta (float | torch.Tensor): rotation angle in radians, arbitrary shape (...,)
        center (torch.Tensor): rotation center, arbitrary shape (..., 2). Default to (0, 0)
        
    Returns:
        (torch.Tensor): (..., 3, 3) transformation matrix
    """
    if isinstance(theta, float):
        theta = torch.tensor(theta)
        if center is not None:
            theta = theta.to(center)
    if center is None:
        center = torch.zeros(2).to(theta).expand(*theta.shape, -1)
    R = rotation_matrix_2d(theta)
    return torch.cat([
        torch.cat([
            R, 
            center[..., :, None] - R @ center[..., :, None],
        ], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=center.dtype, device=center.device).expand(*center.shape[:-1], -1, -1),
    ], dim=-2)


def translate_2d(translation: torch.Tensor):
    """
    Translation matrix for 2D translation
    ```
       [[1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]]
    ```
    Args:
        translation (torch.Tensor): translation vector, arbitrary shape (..., 2)
    
    Returns:
        (torch.Tensor): (..., 3, 3) transformation matrix
    """
    return torch.cat([
        torch.cat([
            torch.eye(2, dtype=translation.dtype, device=translation.device).expand(*translation.shape[:-1], -1, -1),
            translation[..., None],
        ], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=translation.dtype, device=translation.device).expand(*translation.shape[:-1], -1, -1),
    ], dim=-2)


def scale_2d(scale: Union[float, torch.Tensor], center: torch.Tensor = None):
    """
    Scale matrix for 2D scaling
    ```
       [[s, 0, tx],
        [0, s, ty],
        [0, 0,  1]]
    ```
    Args:
        scale (float | torch.Tensor): scale factor, arbitrary shape (...,)
        center (torch.Tensor): scale center, arbitrary shape (..., 2). Default to (0, 0)

    Returns:
        (torch.Tensor): (..., 3, 3) transformation matrix
    """
    if isinstance(scale, float):
        scale = torch.tensor(scale)
        if center is not None:
            scale = scale.to(center)
    if center is None:
        center = torch.zeros(2, dtype=scale.dtype, device=scale.device).expand(*scale.shape, -1)
    return torch.cat([
        torch.cat([
            scale * torch.eye(2, dtype=scale.dtype, device=scale.device).expand(*scale.shape[:-1], -1, -1),
            center[..., :, None] - center[..., :, None] * scale[..., None, None],
        ], dim=-1),
        torch.tensor([[0, 0, 1]], dtype=scale.dtype, device=scale.device).expand(*center.shape[:-1], -1, -1),
    ], dim=-2)


def apply_2d(transform: torch.Tensor, points: torch.Tensor):
    """
    Apply (3x3 or 2x3) 2D affine transformation to points
    ```
        p = R @ p + t
    ```
    Args:
        transform (torch.Tensor): (..., 2 or 3, 3) transformation matrix
        points (torch.Tensor): (..., N, 2) points to transform

    Returns:
        (torch.Tensor): (..., N, 2) transformed points
    """
    assert transform.shape[-2:] == (3, 3) or transform.shape[-2:] == (2, 3), "transform must be 3x3 or 2x3"
    assert points.shape[-1] == 2, "points must be 2D"
    return points @ transform[..., :2, :2].mT + transform[..., :2, None, 2] 