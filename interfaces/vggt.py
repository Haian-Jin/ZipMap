import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import List, Tuple
import time

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import VGGT
from models.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from models.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.utils.geometry import unproject_depth_map_to_point_map
from models.vggt.utils.geometry import closed_form_inverse_se3


def load_and_resize14(filelist: List[str], resize_to: int, device: str):
    images = load_and_preprocess_images(filelist, new_width=resize_to).to(device)

    ori_h, ori_w = images.shape[-2:]
    patch_h, patch_w = ori_h // 14, ori_w // 14
    # (1, 3, h, w) -> (1, 3, h_14, w_14)
    images = F.interpolate(images, (patch_h * 14, patch_w * 14), mode="bilinear", align_corners=False, antialias=True)
    return images


def infer_monodepth(file: str, model: VGGT, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14([file], resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

    depth_map = predictions["depth"]         # (1, 1, h_14, w_14, 1)
    return depth_map[0, 0, ..., 0].detach()  # returns (h_14, w_14) torch tensor


def infer_videodepth(filelist: List[str], model: VGGT, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    
    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)
    end = time.time()

    depth_map = predictions["depth"].squeeze(0).squeeze(-1).cpu()  # depth_map (N, H, W)
    depth_conf = predictions["depth_conf"].squeeze(0).cpu()        # depth_conf (N, H, W)
    return  end - start, depth_map, depth_conf


def infer_cameras_w2c(filelist: List[str], model: VGGT, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    
    h_14, w_14 = images.shape[-2:]
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

    with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
        # (1, N, 3, 4) & (1, N, 3, 3)
        extrinsics_ori, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], image_size_hw=(h_14, w_14))  # h, w only for intrinsics
        extrinsics = torch.eye(4, 4)[None].repeat(extrinsics_ori.shape[1], 1, 1)
        extrinsics[:, :3, :] = extrinsics_ori[0, :, :3, :]

    # since we don't eval intrinsics, just return None
    return extrinsics, None
    # return extrinsics, intrinsics[0]


def infer_cameras_c2w(filelist: List[str], model: VGGT, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)

    h_14, w_14 = images.shape[-2:]
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

    with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
        # (1, N, 3, 4) & (1, N, 3, 3)
        extrinsics_ori, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], image_size_hw=(h_14, w_14))  # h, w only for intrinsics
        extrinsics = torch.eye(4, 4)[None].repeat(extrinsics_ori.shape[1], 1, 1)
        extrinsics[:, :3, :] = extrinsics_ori[0, :, :3, :]

    # Return both extrinsics and intrinsics
    # intrinsics[0] shape: (N, 3, 3), contains fx, fy, cx, cy
    return closed_form_inverse_se3(extrinsics)[:, :3, :], intrinsics[0].detach().cpu()


def infer_mv_pointclouds(filelist: List[str], model: VGGT, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = model(images)

    depth_map = predictions["depth"].squeeze(0).squeeze(-1).unsqueeze(1).cpu()  # -> (N, 1, h, w)
    depth_map = F.interpolate(depth_map, data_size, mode="bilinear", align_corners=False, antialias=True)  # align to gt
    depth_map = depth_map.squeeze(1)  # -> (N, h, w)

    with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
        # (1, N, 3, 4) & (1, N, 3, 3)
        extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions['pose_enc'], image_size_hw=data_size)  # h, w only for intrinsics

    point_map_by_unprojection = unproject_depth_map_to_point_map(
        depth_map.unsqueeze(-1), 
        extrinsics.squeeze(0), 
        intrinsics.squeeze(0),
    )

    return point_map_by_unprojection