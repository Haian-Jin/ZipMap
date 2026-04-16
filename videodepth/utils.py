# Reference: https://github.com/CUT3R/CUT3R/blob/main/eval/video_depth/utils.py

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib as mpl
import imageio.v2 as iio

from typing import Optional
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def save_depth_maps(depth_maps: torch.Tensor, path: str, conf_self: Optional[torch.Tensor] = None):
    min_depth = depth_maps.min()  # float(torch.quantile(out, 0.01))
    max_depth = depth_maps.max()  # float(torch.quantile(out, 0.99))
    
    colored_depth = colorize_optimized(
        depth_maps,
        cmap_name="Spectral_r",
        value_range=(min_depth, max_depth),
        append_cbar=True,
    )

    if conf_self is not None:
        if isinstance(conf_self, list):
            if len(conf_self[0].shape) == 3:
                conf_selfs = torch.cat(conf_self, dim=0)    # (1, H, W) -> (N, H, W)
            elif len(conf_self[0].shape) == 2:
                conf_selfs = torch.stack(conf_self, dim=0)  # (H, W) -> (N, H, W)
        else:
            conf_selfs = conf_self
        log_conf = torch.log(conf_selfs)
        min_conf = log_conf.min()  # float(torch.quantile(out, 0.01))
        max_conf = log_conf.max()  # float(torch.quantile(out, 0.99))
        colored_conf = colorize_optimized(
            log_conf,
            cmap_name="jet",
            value_range=(min_conf, max_conf),
            append_cbar=True,
        )

    img_paths = [f"{path}/frame_{i:04d}.png" for i in range(len(colored_depth))]
    npy_paths = [f"{path}/frame_{i:04d}.npy" for i in range(len(depth_maps))]

    if conf_self is None:
        to_save = (colored_depth * 255).detach().cpu().numpy().astype(np.uint8)
    else:
        to_save = torch.cat([colored_depth, colored_conf], dim=2)  # 沿宽度方向连接
        to_save = (to_save * 255).detach().cpu().numpy().astype(np.uint8)
    
    for i, (img_path, npy_path, img_data) in enumerate(zip(img_paths, npy_paths, to_save)):
        iio.imwrite(img_path, img_data)
        np.save(npy_path, depth_maps[i].detach().cpu().numpy())

    # comment this as it may fail sometimes
    # images = [Image.open(img_path) for img_path in img_paths]
    # images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)

    return depth_maps


def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    cmap = mpl.cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical")
    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]
    cb1.set_ticklabels(tick_label)
    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)
    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()
    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))
    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    return im


def colorize_optimized(
    x, cmap_name="jet", mask=None, value_range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2
):
    device = x.device
    original_shape = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(0)
    B, H, W = x.shape
    
    # deal with vmin/vmax
    if value_range is not None:
        vmin, vmax = value_range
        vmin = torch.full((B,), vmin, device=device)
        vmax = torch.full((B,), vmax, device=device)
    else:
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            mask = mask.expand(B, H, W)
            non_zero_mask = mask & (x != 0)
            has_non_zero = non_zero_mask.reshape(B, -1).any(dim=1)
            
            # min value of non-zero elements in the mask
            non_zero_vmin = x.masked_fill(~non_zero_mask, float('inf')).view(B, -1).min(dim=1)[0]
            # min value of all masked elements
            mask_vmin = x.masked_fill(~mask, float('inf')).view(B, -1).min(dim=1)[0]
            vmin = torch.where(has_non_zero, non_zero_vmin, mask_vmin)
            # set unmasked values -> vmin
            x = x.masked_fill(~mask, vmin.view(B, 1, 1))
            # calculate vmax
            vmax = x.masked_fill(~mask, float('-inf')).view(B, -1).max(dim=1)[0]
        else:
            # if no mask, use quantiles
            x_flatten = x.view(B, -1)
            vmin = torch.quantile(x_flatten, 0.01, dim=1)
            vmax = torch.quantile(x_flatten, 1.0, dim=1) + 1e-6
    
    # clip and normalize the input
    x_clipped = torch.clamp(x, min=vmin.view(B,1,1), max=vmax.view(B,1,1))
    x_normalized = (x_clipped - vmin.view(B,1,1)) / (vmax.view(B,1,1) - vmin.view(B,1,1) + 1e-6)
    
    # generate color map
    cmap = mpl.cm.get_cmap(cmap_name)
    colormap = cmap(np.linspace(0, 1, 256))[:,:3]  # (256,3)
    colormap = torch.from_numpy(colormap).float().to(device)  # (256,3)
    
    # vectorized color mapping
    x_scaled = (x_normalized * 255).long().clamp(0, 255)  # (B,H,W)
    colorized = colormap[x_scaled.flatten()].view(B,H,W,3)  # (B,H,W,3)
    
    # erode the mask
    if mask is not None:
        kernel = torch.ones(3,3, device=device)
        mask_eroded = F.conv2d(
            mask.float().unsqueeze(1), 
            kernel.view(1,1,3,3), 
            padding=1
        ) == kernel.numel()
        mask_eroded = mask_eroded.squeeze(1).unsqueeze(-1)  # (B,H,W,1)
        colorized = colorized * mask_eroded + (1.0 - mask_eroded)
    
    # deal with color bar
    if append_cbar:
        final_images = []
        for i in range(B):
            img_np = colorized[i].detach().cpu().numpy()
            cbar = get_vertical_colorbar(H, vmin[i].item(), vmax[i].item(), cmap_name, cbar_precision=cbar_precision)
            if cbar_in_image:
                cbar_width = cbar.shape[1]
                img_np[:, -cbar_width:] = cbar
            else:
                img_np = np.concatenate([img_np, np.zeros_like(img_np[:,:5]), cbar], axis=1)
            final_images.append(torch.from_numpy(img_np).to(device))
        result = torch.stack(final_images, dim=0)
    else:
        result = colorized.permute(0,3,1,2)  # (B,C,H,W)
    
    # back to original shape if needed
    if len(original_shape) == 2:
        result = result.squeeze(0)
    return result