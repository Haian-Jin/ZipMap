from typing import *
from numbers import Number
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import image_uv


__all__ = [
    'get_rays',
    'get_image_rays',
    'get_mipnerf_cones',
    'volume_rendering',
    'bin_sample',
    'importance_sample',
    'nerf_render_rays',
    'mipnerf_render_rays',
    'nerf_render_view',
    'mipnerf_render_view',
    'InstantNGP',
]


def get_rays(extrinsics: Tensor, intrinsics: Tensor, uv: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        extrinsics: (..., 4, 4) extrinsics matrices.
        intrinsics: (..., 3, 3) intrinsics matrices.
        uv: (..., n_rays, 2) uv coordinates of the rays. 

    Returns:
        rays_o: (..., 1,      3) ray origins
        rays_d: (..., n_rays, 3) ray directions. 
            NOTE: ray directions are NOT normalized. They actuallys makes rays_o + rays_d * z = world coordinates, where z is the depth.
    """
    uvz = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1).to(extrinsics)                                                          # (n_batch, n_views, n_rays, 3)

    with torch.cuda.amp.autocast(enabled=False):
        inv_transformation = (intrinsics @ extrinsics[..., :3, :3]).inverse()
        inv_extrinsics = extrinsics.inverse()
    rays_d = uvz @ inv_transformation.transpose(-1, -2)                                                  
    rays_o = inv_extrinsics[..., None, :3, 3]                                                                                           # (n_batch, n_views, 1, 3)
    return rays_o, rays_d


def get_image_rays(extrinsics: Tensor, intrinsics: Tensor, width: int, height: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        extrinsics: (..., 4, 4) extrinsics matrices.
        intrinsics: (..., 3, 3) intrinsics matrices.
        width: width of the image.
        height: height of the image.
    
    Returns:
        rays_o: (..., 1,      1,     3) ray origins
        rays_d: (..., height, width, 3) ray directions. 
            NOTE: ray directions are NOT normalized. They actuallys makes rays_o + rays_d * z = world coordinates, where z is the depth.
    """
    uv = image_uv(height, width).to(extrinsics).flatten(0, 1)
    rays_o, rays_d = get_rays(extrinsics, intrinsics, uv)
    rays_o = rays_o.unflatten(-2, (1, 1))
    rays_d = rays_d.unflatten(-2, (height, width))
    return rays_o, rays_d


def get_mipnerf_cones(rays_o: Tensor, rays_d: Tensor, z_vals: Tensor, pixel_width: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        rays_o: (..., n_rays, 3) ray origins
        rays_d: (..., n_rays, 3) ray directions.
        z_vals: (..., n_rays, n_samples) z values.
        pixel_width: (...) pixel width. = 1 / (normalized focal length * width)
    
    Returns:
        mu: (..., n_rays, n_samples, 3) cone mu.
        sigma: (..., n_rays, n_samples, 3, 3) cone sigma.
    """
    t_mu = (z_vals[..., 1:] + z_vals[..., :-1]).mul_(0.5)
    t_delta = (z_vals[..., 1:] - z_vals[..., :-1]).mul_(0.5)
    t_mu_square = t_mu.square()
    t_delta_square = t_delta.square()
    t_delta_quad = t_delta_square.square()
    mu_t = t_mu + 2.0 * t_mu * t_delta_square / (3.0 * t_mu_square + t_delta_square)
    sigma_t = t_delta_square / 3.0 - (4.0 / 15.0) * t_delta_quad / (3.0 * t_mu_square + t_delta_square).square() * (12.0 * t_mu_square - t_delta_square)
    sigma_r = (pixel_width[..., None, None].square() / 3.0) * (t_mu_square / 4.0 + (5.0 / 12.0) * t_delta_square - (4.0 / 15.0) * t_delta_quad / (3.0 * t_mu_square + t_delta_square))
    points_mu = rays_o[:, :, :, None, :] + rays_d[:, :, :, None, :] * mu_t[..., None]
    d_dt = rays_d[..., :, None] * rays_d[..., None, :]      # (..., n_rays, 3, 3)
    points_sigma = sigma_t[..., None, None] * d_dt[..., None, :, :] + sigma_r[..., None, None] * (torch.eye(3).to(rays_o) - d_dt[..., None, :, :])
    return points_mu, points_sigma


def get_pixel_width(intrinsics: Tensor, width: int, height: int) -> Tensor:
    """
    Args:
        intrinsics: (..., 3, 3) intrinsics matrices.
        width: width of the image.
        height: height of the image.
    
    Returns:
        pixel_width: (...) pixel width. = 1 / (normalized focal length * width)
    """
    assert width == height, "Currently, only square images are supported."
    pixel_width = torch.reciprocal((intrinsics[..., 0, 0] * intrinsics[..., 1, 1]).sqrt() * width)
    return pixel_width


def volume_rendering(color: Tensor, sigma: Tensor, z_vals: Tensor, ray_length: Tensor, rgb: bool = True, depth: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Given color, sigma and z_vals (linear depth of the sampling points), render the volume.

    NOTE: By default, color and sigma should have one less sample than z_vals, in correspondence with the average value in intervals.
    If queried color are aligned with z_vals, we use trapezoidal rule to calculate the average values in intervals.

    Args:
        color: (..., n_samples or n_samples - 1, 3) color values.
        sigma: (..., n_samples or n_samples - 1) density values.
        z_vals: (..., n_samples) z values.
        ray_length: (...) length of the ray

    Returns:
        rgb: (..., 3) rendered color values.
        depth: (...) rendered depth values.
        weights (..., n_samples) weights.
    """
    dists = (z_vals[..., 1:] - z_vals[..., :-1]) * ray_length[..., None]
    if color.shape[-2] == z_vals.shape[-1]:
        color = (color[..., 1:, :] + color[..., :-1, :]).mul_(0.5)
        sigma = (sigma[..., 1:] + sigma[..., :-1]).mul_(0.5)                                        
    sigma_delta = sigma * dists                                                      
    transparancy = (-torch.cat([torch.zeros_like(sigma_delta[..., :1]), sigma_delta[..., :-1]], dim=-1).cumsum(dim=-1)).exp_()     # First cumsum then exp for numerical stability
    alpha = 1.0 - (-sigma_delta).exp_()                                               
    weights = alpha * transparancy
    if rgb:
        rgb = torch.sum(weights[..., None] * color, dim=-2) if rgb else None        
    if depth:
        z_vals = (z_vals[..., 1:] + z_vals[..., :-1]).mul_(0.5)
        depth = torch.sum(weights * z_vals, dim=-1) / weights.sum(dim=-1).clamp_min_(1e-8) if depth else None            
    return rgb, depth, weights


def neus_volume_rendering(color: Tensor, sdf: Tensor, s: torch.Tensor, z_vals: Tensor = None, rgb: bool = True, depth: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Given color, sdf values and z_vals (linear depth of the sampling points), do volume rendering. (NeuS)

    Args:
        color: (..., n_samples or n_samples - 1, 3) color values.
        sdf: (..., n_samples) sdf values.
        s: (..., n_samples) S values of S-density function in NeuS. The standard deviation of such S-density distribution is 1 / s.
        z_vals: (..., n_samples) z values.
        ray_length: (...) length of the ray

    Returns:
        rgb: (..., 3) rendered color values.
        depth: (...) rendered depth values.
        weights (..., n_samples) weights.
    """

    if color.shape[-2] == z_vals.shape[-1]:
        color = (color[..., 1:, :] + color[..., :-1, :]).mul_(0.5)

    sigmoid_sdf = torch.sigmoid(s * sdf)
    alpha = F.relu(1 - sigmoid_sdf[..., :-1] / sigmoid_sdf[..., :-1])
    transparancy = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), alpha], dim=-1), dim=-1)
    weights = alpha * transparancy

    if rgb:
        rgb = torch.sum(weights[..., None] * color, dim=-2) if rgb else None        
    if depth:
        z_vals = (z_vals[..., 1:] + z_vals[..., :-1]).mul_(0.5)
        depth = torch.sum(weights * z_vals, dim=-1) / weights.sum(dim=-1).clamp_min_(1e-8) if depth else None            
    return rgb, depth, weights


def bin_sample(size: Union[torch.Size, Tuple[int, ...]], n_samples: int, min_value: Number, max_value: Number, spacing: Literal['linear', 'inverse_linear'], dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    """
    Uniformly (or uniformly in inverse space) sample z values in `n_samples` bins in range [min_value, max_value].
    Args:
        size: size of the rays
        n_samples: number of samples to be sampled, also the number of bins
        min_value: minimum value of the range
        max_value: maximum value of the range
        space: 'linear' or 'inverse_linear'. If 'inverse_linear', the sampling is uniform in inverse space.
    
    Returns:
        z_rand: (*size, n_samples) sampled z values, sorted in ascending order.
    """
    if spacing == 'linear':
        pass
    elif spacing == 'inverse_linear':
        min_value = 1.0 / min_value
        max_value = 1.0 / max_value
    bin_length = (max_value - min_value) / n_samples
    z_rand = (torch.rand(*size, n_samples, device=device, dtype=dtype) - 0.5) * bin_length + torch.linspace(min_value + bin_length * 0.5, max_value - bin_length * 0.5, n_samples, device=device, dtype=dtype)   
    if spacing == 'inverse_linear':
        z_rand = 1.0 / z_rand
    return z_rand


def importance_sample(z_vals: Tensor, weights: Tensor, n_samples: int) -> Tuple[Tensor, Tensor]:
    """
    Importance sample z values.

    NOTE: By default, weights should have one less sample than z_vals, in correspondence with the intervals.
    If weights has the same number of samples as z_vals, we use trapezoidal rule to calculate the average weights in intervals.

    Args:
        z_vals: (..., n_rays, n_input_samples) z values, sorted in ascending order.
        weights: (..., n_rays, n_input_samples or n_input_samples - 1) weights.
        n_samples: number of output samples for importance sampling.
    
    Returns:
        z_importance: (..., n_rays, n_samples) importance sampled z values, unsorted.
    """
    if weights.shape[-1] == z_vals.shape[-1]:
        weights = (weights[..., 1:] + weights[..., :-1]).mul_(0.5)
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)                      # (..., n_rays, n_input_samples - 1)
    bins_a, bins_b = z_vals[..., :-1], z_vals[..., 1:]

    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)                          # (..., n_rays, n_input_samples - 1)
    cdf = torch.cumsum(pdf, dim=-1)
    u = torch.rand(*z_vals.shape[:-1], n_samples, device=z_vals.device, dtype=z_vals.dtype)
    
    inds = torch.searchsorted(cdf, u, right=True).clamp(0, cdf.shape[-1] - 1)         # (..., n_rays, n_samples)
    
    bins_a = torch.gather(bins_a, dim=-1, index=inds)
    bins_b = torch.gather(bins_b, dim=-1, index=inds)
    z_importance = bins_a + (bins_b - bins_a) * torch.rand_like(u)
    return z_importance


def nerf_render_rays(
    nerf: Union[Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]], Tuple[Callable[[Tensor], Tuple[Tensor, Tensor]], Callable[[Tensor], Tuple[Tensor, Tensor]]]],
    rays_o: Tensor, rays_d: Tensor,
    *, 
    return_dict: bool = False,
    n_coarse: int = 64, n_fine: int = 64,
    near: float = 0.1, far: float = 100.0,
    z_spacing: Literal['linear', 'inverse_linear'] = 'linear',
):
    """
    NeRF rendering of rays. Note that it supports arbitrary batch dimensions (denoted as `...`)

    Args:
        nerf: nerf model, which takes (points, directions) as input and returns (color, density) as output.
            If nerf is a tuple, it should be (nerf_coarse, nerf_fine), where nerf_coarse and nerf_fine are two nerf models for coarse and fine stages respectively.
            
            nerf args:
                points: (..., n_rays, n_samples, 3)
                directions: (..., n_rays, n_samples, 3)
            nerf returns:
                color: (..., n_rays, n_samples, 3) color values.
                density: (..., n_rays, n_samples) density values.
                
        rays_o: (..., n_rays, 3) ray origins
        rays_d: (..., n_rays, 3) ray directions.
        pixel_width: (..., n_rays) pixel width. How to compute? pixel_width = 1 / (normalized focal length * width)
    
    Returns 
        if return_dict is False, return rendered rgb and depth for short cut. (If there are separate coarse and fine results, return fine results)
            rgb: (..., n_rays, 3) rendered color values. 
            depth: (..., n_rays) rendered depth values.
        else, return a dict. If `n_fine == 0` or `nerf` is a single model, the dict only contains coarse results:
        ```
        {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
        ```
        If there are two models for coarse and fine stages, the dict contains both coarse and fine results:
        ```
        {
            "coarse": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..},
            "fine": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
        }
        ```
    """
    if isinstance(nerf, tuple):
        nerf_coarse, nerf_fine = nerf
    else:
        nerf_coarse = nerf_fine = nerf
    # 1. Coarse: bin sampling
    z_coarse = bin_sample(rays_d.shape[:-1], n_coarse, near, far, device=rays_o.device, dtype=rays_o.dtype, spacing=z_spacing)                       # (n_batch, n_views, n_rays, n_samples)
    points_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_coarse[..., None]                                                                # (n_batch, n_views, n_rays, n_samples, 3)
    ray_length = rays_d.norm(dim=-1)

    #    Query color and density                   
    color_coarse, density_coarse = nerf_coarse(points_coarse, rays_d[..., None, :].expand_as(points_coarse))               # (n_batch, n_views, n_rays, n_samples, 3), (n_batch, n_views, n_rays, n_samples)
    
    #    Volume rendering
    with torch.no_grad():
        rgb_coarse, depth_coarse, weights = volume_rendering(color_coarse, density_coarse, z_coarse, ray_length)            # (n_batch, n_views, n_rays, 3), (n_batch, n_views, n_rays, 1), (n_batch, n_views, n_rays, n_samples)
    
    if n_fine == 0:
        if return_dict:
            return {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights, 'z_vals': z_coarse, 'color': color_coarse, 'density': density_coarse}
        else:
            return rgb_coarse, depth_coarse
    
    # 2. Fine: Importance sampling
    if nerf_coarse is nerf_fine:
        # If coarse and fine stages share the same model, the points of coarse stage can be reused, 
        # and we only need to query the importance samples of fine stage.
        z_fine = importance_sample(z_coarse, weights, n_fine)               
        points_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_fine[..., None]                      
        color_fine, density_fine = nerf_fine(points_fine, rays_d[..., None, :].expand_as(points_fine))

        # Merge & volume rendering
        z_vals = torch.cat([z_coarse, z_fine], dim=-1)          
        color = torch.cat([color_coarse, color_fine], dim=-2)
        density = torch.cat([density_coarse, density_fine], dim=-1)     
        z_vals, sort_inds = torch.sort(z_vals, dim=-1)                   
        color = torch.gather(color, dim=-2, index=sort_inds[..., None].expand_as(color))
        density = torch.gather(density, dim=-1, index=sort_inds)
        rgb, depth, weights = volume_rendering(color, density, z_vals, ray_length)
        
        if return_dict:
            return {'rgb': rgb, 'depth': depth, 'weights': weights, 'z_vals': z_vals, 'color': color, 'density': density}
        else:
            return rgb, depth
    else:
        # If coarse and fine stages use different models, we need to query the importance samples of both stages.
        z_fine = importance_sample(z_coarse, weights, n_fine)
        z_vals = torch.cat([z_coarse, z_fine], dim=-1)   
        points = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., None]
        color, density = nerf_fine(points)
        rgb, depth, weights = volume_rendering(color, density, z_vals, ray_length)

        if return_dict:
            return {
                'coarse': {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights, 'z_vals': z_coarse, 'color': color_coarse, 'density': density_coarse},
                'fine': {'rgb': rgb, 'depth': depth, 'weights': weights, 'z_vals': z_vals, 'color': color, 'density': density}
            }
        else:
            return rgb, depth


def mipnerf_render_rays(
    mipnerf: Callable[[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]],
    rays_o: Tensor, rays_d: Tensor, pixel_width: Tensor, 
    *, 
    return_dict: bool = False,
    n_coarse: int = 64, n_fine: int = 64, uniform_ratio: float = 0.4,
    near: float = 0.1, far: float = 100.0,
    z_spacing: Literal['linear', 'inverse_linear'] = 'linear',
) -> Union[Tuple[Tensor, Tensor], Dict[str, Tensor]]:
    """
    MipNeRF rendering.

    Args:
        mipnerf: mipnerf model, which takes (points_mu, points_sigma) as input and returns (color, density) as output.

            mipnerf args:
                points_mu: (..., n_rays, n_samples, 3) cone mu.
                points_sigma: (..., n_rays, n_samples, 3, 3) cone sigma.
                directions: (..., n_rays, n_samples, 3)
            mipnerf returns:
                color: (..., n_rays, n_samples, 3) color values.
                density: (..., n_rays, n_samples) density values.

        rays_o: (..., n_rays, 3) ray origins
        rays_d: (..., n_rays, 3) ray directions.
        pixel_width: (..., n_rays) pixel width. How to compute? pixel_width = 1 / (normalized focal length * width)
    
    Returns 
        if return_dict is False, return rendered results only: (If `n_fine == 0`, return coarse results, otherwise return fine results)
            rgb: (..., n_rays, 3) rendered color values. 
            depth: (..., n_rays) rendered depth values.
        else, return a dict. If `n_fine == 0`, the dict only contains coarse results:
        ```
        {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
        ```
        If n_fine > 0, the dict contains both coarse and fine results :
        ```
        {
            "coarse": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..},
            "fine": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
        }
        ```
    """
    # 1. Coarse: bin sampling
    z_coarse = bin_sample(rays_d.shape[:-1], n_coarse, near, far, spacing=z_spacing, device=rays_o.device, dtype=rays_o.dtype)
    points_mu_coarse, points_sigma_coarse = get_mipnerf_cones(rays_o, rays_d, z_coarse, pixel_width)
    ray_length = rays_d.norm(dim=-1)

    #    Query color and density
    color_coarse, density_coarse = mipnerf(points_mu_coarse, points_sigma_coarse, rays_d[..., None, :].expand_as(points_mu_coarse))             # (n_batch, n_views, n_rays, n_samples, 3), (n_batch, n_views, n_rays, n_samples)

    #    Volume rendering
    rgb_coarse, depth_coarse, weights_coarse = volume_rendering(color_coarse, density_coarse, z_coarse, ray_length)                             # (n_batch, n_views, n_rays, 3), (n_batch, n_views, n_rays, 1), (n_batch, n_views, n_rays, n_samples)

    if n_fine == 0:
        if return_dict:
            return {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights_coarse, 'z_vals': z_coarse, 'color': color_coarse, 'density': density_coarse}
        else:
            return rgb_coarse, depth_coarse

    # 2. Fine: Importance sampling. (NOTE: coarse stages and fine stages always share the same model, but coarse stage points can not be reused)
    with torch.no_grad():
        weights_coarse = (1.0 - uniform_ratio) * weights_coarse + uniform_ratio / weights_coarse.shape[-1]
    z_fine = importance_sample(z_coarse, weights_coarse, n_fine)
    z_fine, _ = torch.sort(z_fine, dim=-2)
    points_mu_fine, points_sigma_fine = get_mipnerf_cones(rays_o, rays_d, z_fine, pixel_width)                                                           
    color_fine, density_fine = mipnerf(points_mu_fine, points_sigma_fine, rays_d[..., None, :].expand_as(points_mu_fine))

    #   Volume rendering                    
    rgb_fine, depth_fine, weights_fine = volume_rendering(color_fine, density_fine, z_fine, ray_length)

    if return_dict:
        return {
            'coarse': {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights_coarse, 'z_vals': z_coarse, 'color': color_coarse, 'density': density_coarse},
            'fine': {'rgb': rgb_fine, 'depth': depth_fine, 'weights': weights_fine, 'z_vals': z_fine, 'color': color_fine, 'density': density_fine}
        }
    else:
        return rgb_fine, depth_fine


def neus_render_rays(
    neus: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    s: Union[Number, Tensor],
    rays_o: Tensor, rays_d: Tensor, 
    *, 
    compute_normal: bool = True,
    return_dict: bool = False,
    n_coarse: int = 64, n_fine: int = 64,
    near: float = 0.1, far: float = 100.0,
    z_spacing: Literal['linear', 'inverse_linear'] = 'linear',
):
    """
    TODO
    NeuS rendering of rays. Note that it supports arbitrary batch dimensions (denoted as `...`)

    Args:
        neus: neus model, which takes (points, directions) as input and returns (color, density) as output.

            nerf args:
                points: (..., n_rays, n_samples, 3)
                directions: (..., n_rays, n_samples, 3)
            nerf returns:
                color: (..., n_rays, n_samples, 3) color values.
                density: (..., n_rays, n_samples) density values.
                
        rays_o: (..., n_rays, 3) ray origins
        rays_d: (..., n_rays, 3) ray directions.
        pixel_width: (..., n_rays) pixel width. How to compute? pixel_width = 1 / (normalized focal length * width)
    
    Returns 
        if return_dict is False, return rendered results only: (If `n_fine == 0`, return coarse results, otherwise return fine results)
            rgb: (..., n_rays, 3) rendered color values. 
            depth: (..., n_rays) rendered depth values.
        else, return a dict. If `n_fine == 0`, the dict only contains coarse results:
        ```
        {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'sdf': ..., 'normal': ...}
        ```
        If n_fine > 0, the dict contains both coarse and fine results:
        ```
        {
            "coarse": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..},
            "fine": {'rgb': .., 'depth': .., 'weights': .., 'z_vals': .., 'color': .., 'density': ..}
        }
        ```
    """

    # 1. Coarse: bin sampling
    z_coarse = bin_sample(rays_d.shape[:-1], n_coarse, near, far, device=rays_o.device, dtype=rays_o.dtype, spacing=z_spacing)                       # (n_batch, n_views, n_rays, n_samples)
    points_coarse = rays_o[..., None, :] + rays_d[..., None, :] * z_coarse[..., None]                                                                # (n_batch, n_views, n_rays, n_samples, 3)

    #    Query color and density                   
    color_coarse, sdf_coarse = neus(points_coarse, rays_d[..., None, :].expand_as(points_coarse))                                  # (n_batch, n_views, n_rays, n_samples, 3), (n_batch, n_views, n_rays, n_samples)
    
    #    Volume rendering
    with torch.no_grad():
        rgb_coarse, depth_coarse, weights = neus_volume_rendering(color_coarse, sdf_coarse, s, z_coarse)            # (n_batch, n_views, n_rays, 3), (n_batch, n_views, n_rays, 1), (n_batch, n_views, n_rays, n_samples)
    
    if n_fine == 0:
        if return_dict:
            return {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights, 'z_vals': z_coarse, 'color': color_coarse, 'sdf': sdf_coarse}
        else:
            return rgb_coarse, depth_coarse
    
    # If coarse and fine stages share the same model, the points of coarse stage can be reused, 
    # and we only need to query the importance samples of fine stage.
    z_fine = importance_sample(z_coarse, weights, n_fine)               
    points_fine = rays_o[..., None, :] + rays_d[..., None, :] * z_fine[..., None]                      
    color_fine, sdf_fine = neus(points_fine, rays_d[..., None, :].expand_as(points_fine))

    # Merge & volume rendering
    z_vals = torch.cat([z_coarse, z_fine], dim=-1)          
    color = torch.cat([color_coarse, color_fine], dim=-2)
    sdf = torch.cat([sdf_coarse, sdf_fine], dim=-1)     
    z_vals, sort_inds = torch.sort(z_vals, dim=-1)                   
    color = torch.gather(color, dim=-2, index=sort_inds[..., None].expand_as(color))
    sdf = torch.gather(sdf, dim=-1, index=sort_inds)
    rgb, depth, weights = neus_volume_rendering(color, sdf, s, z_vals)

    if return_dict:
        return {
            'coarse': {'rgb': rgb_coarse, 'depth': depth_coarse, 'weights': weights, 'z_vals': z_coarse, 'color': color_coarse, 'sdf': sdf_coarse},
            'fine': {'rgb': rgb, 'depth': depth, 'weights': weights, 'z_vals': z_vals, 'color': color, 'sdf': sdf}
        }
    else:
        return rgb, depth


def nerf_render_view(
    nerf: Tensor,
    extrinsics: Tensor, 
    intrinsics: Tensor, 
    width: int,
    height: int,
    *,
    patchify: bool = False,
    patch_size: Tuple[int, int] = (64, 64),
    **options: Dict[str, Any]
) -> Tuple[Tensor, Tensor]:
    """
    NeRF rendering of views. Note that it supports arbitrary batch dimensions (denoted as `...`)

    Args:
        extrinsics: (..., 4, 4) extrinsics matrice of the rendered views
        intrinsics (optional): (..., 3, 3) intrinsics matrice of the rendered views.
        width (optional): image width of the rendered views.
        height (optional): image height of the rendered views.
        patchify (optional): If the image is too large, render it patch by patch
        **options: rendering options.
    
    Returns:
        rgb: (..., channels, height, width) rendered color values.
        depth: (..., height, width) rendered depth values.
    """
    if patchify:
        # Patchified rendering
        max_patch_width, max_patch_height = patch_size
        n_rows, n_columns = math.ceil(height / max_patch_height), math.ceil(width / max_patch_width)

        rgb_rows, depth_rows = [], []
        for i_row in range(n_rows):
            rgb_row, depth_row = [], []
            for i_column in range(n_columns):
                patch_shape = patch_height, patch_width = min(max_patch_height, height - i_row * max_patch_height), min(max_patch_width, width - i_column * max_patch_width)
                uv = image_uv(height, width, i_column * max_patch_width, i_row * max_patch_height, i_column * max_patch_width + patch_width, i_row * max_patch_height + patch_height).to(extrinsics)
                uv = uv.flatten(0, 1)                                               # (patch_height * patch_width, 2)
                ray_o_, ray_d_ = get_rays(extrinsics, intrinsics, uv)
                rgb_, depth_ = nerf_render_rays(nerf, ray_o_, ray_d_, **options, return_dict=False)
                rgb_ = rgb_.transpose(-1, -2).unflatten(-1, patch_shape)            # (..., 3, patch_height, patch_width)
                depth_ = depth_.unflatten(-1, patch_shape)                          # (..., patch_height, patch_width)
                
                rgb_row.append(rgb_)
                depth_row.append(depth_)
            rgb_rows.append(torch.cat(rgb_row, dim=-1))
            depth_rows.append(torch.cat(depth_row, dim=-1))
        rgb = torch.cat(rgb_rows, dim=-2)
        depth = torch.cat(depth_rows, dim=-2)

        return rgb, depth
    else:
        # Full rendering
        uv = image_uv(height, width).to(extrinsics)
        uv = uv.flatten(0, 1)                                                       # (height * width, 2)
        ray_o_, ray_d_ = get_rays(extrinsics, intrinsics, uv)
        rgb, depth = nerf_render_rays(nerf, ray_o_, ray_d_, **options, return_dict=False)
        rgb = rgb.transpose(-1, -2).unflatten(-1, (height, width))                  # (..., 3, height, width)
        depth = depth.unflatten(-1, (height, width))                                # (..., height, width)
        
        return rgb, depth
    

def mipnerf_render_view(
    mipnerf: Tensor,
    extrinsics: Tensor, 
    intrinsics: Tensor, 
    width: int,
    height: int,
    *,
    patchify: bool = False,
    patch_size: Tuple[int, int] = (64, 64),
    **options: Dict[str, Any]
) -> Tuple[Tensor, Tensor]:
    """
    MipNeRF rendering of views. Note that it supports arbitrary batch dimensions (denoted as `...`)

    Args:
        extrinsics: (..., 4, 4) extrinsics matrice of the rendered views
        intrinsics (optional): (..., 3, 3) intrinsics matrice of the rendered views.
        width (optional): image width of the rendered views.
        height (optional): image height of the rendered views.
        patchify (optional): If the image is too large, render it patch by patch
        **options: rendering options.
    
    Returns:
        rgb: (..., 3, height, width) rendered color values.
        depth: (..., height, width) rendered depth values.
    """
    pixel_width = get_pixel_width(intrinsics, width, height)

    if patchify:
        # Patchified rendering
        max_patch_width, max_patch_height = patch_size
        n_rows, n_columns = math.ceil(height / max_patch_height), math.ceil(width / max_patch_width)

        rgb_rows, depth_rows = [], []
        for i_row in range(n_rows):
            rgb_row, depth_row = [], []
            for i_column in range(n_columns):
                patch_shape = patch_height, patch_width = min(max_patch_height, height - i_row * max_patch_height), min(max_patch_width, width - i_column * max_patch_width)
                uv = image_uv(height, width, i_column * max_patch_width, i_row * max_patch_height, i_column * max_patch_width + patch_width, i_row * max_patch_height + patch_height).to(extrinsics)
                uv = uv.flatten(0, 1)                                               # (patch_height * patch_width, 2)
                ray_o_, ray_d_ = get_rays(extrinsics, intrinsics, uv)
                rgb_, depth_ = mipnerf_render_rays(mipnerf, ray_o_, ray_d_, pixel_width, **options) 
                rgb_ = rgb_.transpose(-1, -2).unflatten(-1, patch_shape)            # (..., 3, patch_height, patch_width)
                depth_ = depth_.unflatten(-1, patch_shape)                          # (..., patch_height, patch_width)
                
                rgb_row.append(rgb_)
                depth_row.append(depth_)
            rgb_rows.append(torch.cat(rgb_row, dim=-1))
            depth_rows.append(torch.cat(depth_row, dim=-1))
        rgb = torch.cat(rgb_rows, dim=-2)
        depth = torch.cat(depth_rows, dim=-2)

        return rgb, depth
    else:
        # Full rendering
        uv = image_uv(height, width).to(extrinsics)
        uv = uv.flatten(0, 1)                                                       # (height * width, 2)
        ray_o_, ray_d_ = get_rays(extrinsics, intrinsics, uv)
        rgb, depth = mipnerf_render_rays(mipnerf, ray_o_, ray_d_, pixel_width, **options) 
        rgb = rgb.transpose(-1, -2).unflatten(-1, (height, width))                  # (..., 3, height, width)
        depth = depth.unflatten(-1, (height, width))                                # (..., height, width)
        
        return rgb, depth


class InstantNGP(nn.Module):
    """
    An implementation of InstantNGP, Müller et. al., https://nvlabs.github.io/instant-ngp/.
    Requires `tinycudann` package.
    Install it by:
    ```
    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
    ```
    """
    def __init__(self,
        view_dependent: bool = True,
        base_resolution: int = 16,
        finest_resolution: int = 2048,
        n_levels: int = 16,
        num_layers_density: int = 2,
        hidden_dim_density: int = 64,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        log2_hashmap_size: int = 19,
        bound: float = 1.0,
        color_channels: int = 3,
    ):
        super().__init__()
        import tinycudann
        N_FEATURES_PER_LEVEL = 2
        GEO_FEAT_DIM = 15

        self.bound = bound
        self.color_channels = color_channels

        # density network
        self.num_layers_density = num_layers_density
        self.hidden_dim_density = hidden_dim_density

        per_level_scale = (finest_resolution / base_resolution) ** (1 / (n_levels - 1))

        self.encoder = tinycudann.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": N_FEATURES_PER_LEVEL,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )

        self.density_net = tinycudann.Network(
            n_input_dims=N_FEATURES_PER_LEVEL * n_levels,
            n_output_dims=1 + GEO_FEAT_DIM,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_density,
                "n_hidden_layers": num_layers_density - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        
        self.view_dependent = view_dependent
        if view_dependent:
            self.encoder_dir = tinycudann.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
            self.in_dim_color = self.encoder_dir.n_output_dims + GEO_FEAT_DIM
        else:
            self.in_dim_color = GEO_FEAT_DIM

        self.color_net = tinycudann.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=color_channels,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )
    
    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (..., 3) points
            d: (..., 3) directions
        Returns:
            color: (..., 3) color values.
            density: (..., 1) density values.
        """
        batch_shape = x.shape[:-1]
        x, d = x.reshape(-1, 3), d.reshape(-1, 3)

        # density
        x = (x + self.bound) / (2 * self.bound)     # to [0, 1]
        x = self.encoder(x)
        density, geo_feat = self.density_net(x).split([1, 15], dim=-1)
        density = F.softplus(density).squeeze(-1)

        # color
        if self.view_dependent:
            d = (F.normalize(d, dim=-1) + 1) / 2    # tcnn SH encoding requires inputs to be in [0, 1]
            d = self.encoder_dir(d)
            h = torch.cat([d, geo_feat], dim=-1)
        else:
            h = geo_feat
        color = self.color_net(h)

        return color.reshape(*batch_shape, self.color_channels), density.reshape(*batch_shape)

