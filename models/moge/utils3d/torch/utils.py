from typing import *

import torch
import torch.nn.functional as F

from . import transforms
from . import mesh
from ._helpers import batched
from .._helpers import no_warnings


__all__ = [
    'sliding_window_1d',
    'sliding_window_2d',
    'sliding_window_nd',
    'image_uv',
    'image_pixel_center',
    'image_mesh',
    'chessboard',
    'depth_edge',
    'depth_aliasing',
    'image_mesh_from_depth',
    'points_to_normals',
    'depth_to_points',
    'depth_to_normals',
    'masked_min',
    'masked_max',
    'bounding_rect'
]


def sliding_window_1d(x: torch.Tensor, window_size: int, stride: int = 1, dim: int = -1) -> torch.Tensor:
    """
    Sliding window view of the input tensor. The dimension of the sliding window is appended to the end of the input tensor's shape.
    NOTE: Since Pytorch has `unfold` function, 1D sliding window view is just a wrapper of it.
    """
    return x.unfold(dim, window_size, stride)


def sliding_window_nd(x: torch.Tensor, window_size: Tuple[int, ...], stride: Tuple[int, ...], dim: Tuple[int, ...]) -> torch.Tensor:
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    assert len(window_size) == len(stride) == len(dim)
    for i in range(len(window_size)):
        x = sliding_window_1d(x, window_size[i], stride[i], dim[i])
    return x


def sliding_window_2d(x: torch.Tensor, window_size: Union[int, Tuple[int, int]], stride: Union[int, Tuple[int, int]], dim: Union[int, Tuple[int, int]] = (-2, -1)) -> torch.Tensor:
    if isinstance(window_size, int):
        window_size = (window_size, window_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    return sliding_window_nd(x, window_size, stride, dim)


def image_uv(height: int, width: int, left: int = None, top: int = None, right: int = None, bottom: int = None, device: torch.device = None, dtype: torch.dtype = None) -> torch.Tensor:
    """
    Get image space UV grid, ranging in [0, 1]. 

    >>> image_uv(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace((left + 0.5) / width, (right - 0.5) / width, right - left, device=device, dtype=dtype)
    v = torch.linspace((top + 0.5) / height, (bottom - 0.5) / height, bottom - top, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def image_pixel_center(
    height: int,
    width: int,
    left: int = None,
    top: int = None,
    right: int = None,
    bottom: int = None,
    dtype: torch.dtype = None,
    device: torch.device = None
) -> torch.Tensor:
    """
    Get image pixel center coordinates, ranging in [0, width] and [0, height].
    `image[i, j]` has pixel center coordinates `(j + 0.5, i + 0.5)`.

    >>> image_pixel_center(10, 10):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]

    Args:
        width (int): image width
        height (int): image height

    Returns:
        torch.Tensor: shape (height, width, 2)
    """
    if left is None: left = 0
    if top is None: top = 0
    if right is None: right = width
    if bottom is None: bottom = height
    u = torch.linspace(left + 0.5, right - 0.5, right - left, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5, bottom - 0.5, bottom - top, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


def image_mesh(height: int, width: int, mask: torch.Tensor = None, device: torch.device = None, dtype: torch.dtype = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a quad mesh regarding image pixel uv coordinates as vertices and image grid as faces.

    Args:
        width (int): image width
        height (int): image height
        mask (torch.Tensor, optional): binary mask of shape (height, width), dtype=bool. Defaults to None.

    Returns:
        uv (torch.Tensor): uv corresponding to pixels as described in image_uv()
        faces (torch.Tensor): quad faces connecting neighboring pixels
        indices (torch.Tensor, optional): indices of vertices in the original mesh
    """
    if device is None and mask is not None:
        device = mask.device
    if mask is not None:
        assert mask.shape[0] == height and mask.shape[1] == width
        assert mask.dtype == torch.bool
    uv = image_uv(height, width, device=device, dtype=dtype).reshape((-1, 2))
    row_faces = torch.stack([
        torch.arange(0, width - 1, dtype=torch.int32, device=device), 
        torch.arange(width, 2 * width - 1, dtype=torch.int32, device=device), 
        torch.arange(1 + width, 2 * width, dtype=torch.int32, device=device), 
        torch.arange(1, width, dtype=torch.int32, device=device)
    ], dim=1)
    faces = (torch.arange(0, (height - 1) * width, width, device=device, dtype=torch.int32)[:, None, None] + row_faces[None, :, :]).reshape((-1, 4))
    if mask is not None:
        quad_mask = (mask[:-1, :-1] & mask[1:, :-1] & mask[1:, 1:] & mask[:-1, 1:]).ravel()
        faces = faces[quad_mask]
        faces, uv, indices = mesh.remove_unreferenced_vertices(faces, uv, return_indices=True)
        return uv, faces, indices
    return uv, faces


def depth_edge(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the edge mask of a depth map. The edge is defined as the pixels whose neighbors have a large difference in depth.
    
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff = (F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2))
    else:
        diff = (F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2))

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


def depth_aliasing(depth: torch.Tensor, atol: float = None, rtol: float = None, kernel_size: int = 3, mask: torch.Tensor = None) -> torch.BoolTensor:
    """
    Compute the map that indicates the aliasing of a depth map. The aliasing is defined as the pixels which neither close to the maximum nor the minimum of its neighbors.
    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        atol (float): absolute tolerance
        rtol (float): relative tolerance

    Returns:
        edge (torch.Tensor): shape (..., height, width) of dtype torch.bool
    """
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])

    if mask is None:
        diff_max = F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2) + depth
    else:
        diff_max = F.max_pool2d(torch.where(mask, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) - depth
        diff_min = F.max_pool2d(torch.where(mask, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2) + depth
    diff = torch.minimum(diff_max, diff_min)

    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    edge = edge.reshape(*shape)
    return edge


def image_mesh_from_depth(
    depth: torch.Tensor,
    extrinsics: torch.Tensor = None,
    intrinsics: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    height, width = depth.shape
    uv, faces = image_mesh(height, width)
    faces = faces.reshape(-1, 4)
    depth = depth.reshape(-1)
    pts = transforms.unproject_cv(image_uv, depth, extrinsics, intrinsics)
    faces = mesh.triangulate(faces, vertices=pts)
    return pts, faces


@batched(3, 2, 2)
def points_to_normals(point: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate normal map from point map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

    Args:
        point (torch.Tensor): shape (..., height, width, 3), point map
    Returns:
        normal (torch.Tensor): shape (..., height, width, 3), normal map. 
    """
    has_mask = mask is not None

    if mask is None:
        mask = torch.ones_like(point[..., 0], dtype=torch.bool)
    mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)

    pts = F.pad(point.permute(0, 3, 1, 2), (1, 1, 1, 1), mode='constant', value=1).permute(0, 2, 3, 1)
    up = pts[:, :-2, 1:-1, :] - pts[:, 1:-1, 1:-1, :]
    left = pts[:, 1:-1, :-2, :] - pts[:, 1:-1, 1:-1, :]
    down = pts[:, 2:, 1:-1, :] - pts[:, 1:-1, 1:-1, :]
    right = pts[:, 1:-1, 2:, :] - pts[:, 1:-1, 1:-1, :]
    normal = torch.stack([
        torch.cross(up, left, dim=-1),
        torch.cross(left, down, dim=-1),
        torch.cross(down, right, dim=-1),
        torch.cross(right, up, dim=-1),
    ])
    normal = F.normalize(normal, dim=-1)
    valid = torch.stack([
        mask[:, :-2, 1:-1] & mask[:, 1:-1, :-2],
        mask[:, 1:-1, :-2] & mask[:, 2:, 1:-1],
        mask[:, 2:, 1:-1] & mask[:, 1:-1, 2:],
        mask[:, 1:-1, 2:] & mask[:, :-2, 1:-1],
    ]) & mask[None, :, 1:-1, 1:-1]
    normal = (normal * valid[..., None]).sum(dim=0)
    normal = F.normalize(normal, dim=-1)
    
    if has_mask:
        return normal, valid.any(dim=0)
    else:
        return normal


def depth_to_normals(depth: torch.Tensor, intrinsics: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Calculate normal map from depth map. Value range is [-1, 1]. Normal direction in OpenGL identity camera's coordinate system.

    Args:
        depth (torch.Tensor): shape (..., height, width), linear depth map
        intrinsics (torch.Tensor): shape (..., 3, 3), intrinsics matrix
    Returns:
        normal (torch.Tensor): shape (..., 3, height, width), normal map. 
    """
    pts = depth_to_points(depth, intrinsics)
    return points_to_normals(pts, mask)


def depth_to_points(depth: torch.Tensor, intrinsics: torch.Tensor, extrinsics: torch.Tensor = None):
    height, width = depth.shape[-2:]
    uv = image_uv(width=width, height=height, dtype=depth.dtype, device=depth.device)
    pts = transforms.unproject_cv(uv, depth, intrinsics=intrinsics[..., None, :, :], extrinsics=extrinsics[..., None, :, :] if extrinsics is not None else None)
    return pts


def masked_min(input: torch.Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Similar to torch.min, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min()
    else:
        return torch.where(mask, input, torch.tensor(torch.inf, dtype=input.dtype, device=input.device)).min(dim=dim, keepdim=keepdim)


def masked_max(input: torch.Tensor, mask: torch.BoolTensor, dim: int = None, keepdim: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Similar to torch.max, but with mask
    """
    if dim is None:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max()
    else:
        return torch.where(mask, input, torch.tensor(-torch.inf, dtype=input.dtype, device=input.device)).max(dim=dim, keepdim=keepdim)
    

def bounding_rect(mask: torch.BoolTensor):
    """get bounding rectangle of a mask

    Args:
        mask (torch.Tensor): shape (..., height, width), mask

    Returns:
        rect (torch.Tensor): shape (..., 4), bounding rectangle (left, top, right, bottom)
    """
    height, width = mask.shape[-2:]
    mask = mask.flatten(-2).unsqueeze(-1)
    uv = image_uv(height, width).to(mask.device).reshape(-1, 2)
    left_top = masked_min(uv, mask, dim=-2)[0]
    right_bottom = masked_max(uv, mask, dim=-2)[0]
    return torch.cat([left_top, right_bottom], dim=-1)


def chessboard(width: int, height: int, grid_size: int, color_a: torch.Tensor, color_b: torch.Tensor) -> torch.Tensor:
    """get a chessboard image

    Args:
        width (int): image width
        height (int): image height
        grid_size (int): size of chessboard grid
        color_a (torch.Tensor): shape (chanenls,), color of the grid at the top-left corner
        color_b (torch.Tensor): shape (chanenls,), color in complementary grids

    Returns:
        image (torch.Tensor): shape (height, width, channels), chessboard image
    """
    x = torch.div(torch.arange(width), grid_size, rounding_mode='floor')
    y = torch.div(torch.arange(height), grid_size, rounding_mode='floor')
    mask = ((x[None, :] + y[:, None]) % 2).to(color_a)
    image = (1 - mask[..., None]) * color_a + mask[..., None] * color_b
    return image