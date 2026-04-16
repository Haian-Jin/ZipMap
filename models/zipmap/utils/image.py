import torch
import numpy as np
import matplotlib
import logging
from typing import List

def depth_to_np_arr(depth):
    '''
    visualizing depth map
        input: 
            depth maps of one scene, (S, H, W)
        output: 
            a list of normalized and colormapped depth maps of one scene
            each element is a numpy array of shape (H, W, 3)
    '''


    if isinstance(depth, list):
        depth = torch.stack(depth)
    depth = depth.detach().cpu().float().numpy()
    max_depth = np.percentile(depth, 98)
    min_depth = np.percentile(depth, 2)

    depth = (depth - min_depth) / (max_depth - min_depth) 
    depth = np.clip(depth, 0.0, 1.0)

    # cmap = matplotlib.colormaps.get_cmap('inferno')
    cmap = matplotlib.colormaps.get_cmap('turbo')

    np_depth = []
    for i in range(len(depth)):
        d = depth[i]
        if d.min() == d.max():
            logging.info(f"Depth min and max are the same for {i}")
            d = np.zeros_like(d)
        image = cmap(d)[:, :, :3] * 255
        np_depth.append(image.astype(np.uint8))

    return np_depth



def stack_images(image_rows: List[List[np.ndarray]], spacing=5, spacing_color=0) -> np.ndarray:
    '''
    Efficiently stack a list of images into a grid with optional spacing.
    input: a list of list, each element is a list of images representing a row, each image is a numpy array of shape (H, W, 3)
    output: a numpy array of shape (H*nrow + spacing*(nrow-1), W*ncol + spacing*(ncol-1), 3)
    '''
    nrow = len(image_rows)

    ncol = len(image_rows[0])
    H, W = image_rows[0][0].shape[:2]

    # Calculate the final grid size, only add spacing between images, not at the edges
    grid_height = nrow * H + spacing * (nrow - 1)
    grid_width = ncol * W + spacing * (ncol - 1)
    image_grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * spacing_color
    for i in range(nrow):
        for j in range(ncol):
            y = i * (H + spacing)
            x = j * (W + spacing)
            image_grid[y:y+H, x:x+W] = image_rows[i][j]
    return image_grid