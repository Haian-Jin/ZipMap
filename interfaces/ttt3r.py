import os
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from typing import List, Tuple

import imageio.v3 as iio
from copy import deepcopy

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import TTT3R
from models.vggt.utils.load_fn import load_and_preprocess_images
from models.vggt.utils.geometry import closed_form_inverse_se3


def load_and_resize16(filelist: List[str], resize_to: int, device: str):
    images = load_and_preprocess_images(filelist, new_width=resize_to).to(device)

    ori_h, ori_w = images.shape[-2:]
    patch_h, patch_w = ori_h // 16, ori_w // 16
    # (1, 3, h, w) -> (1, 3, h_16, w_16)
    images = F.interpolate(images, (patch_h * 16, patch_w * 16), mode="bilinear", align_corners=False, antialias=True)
    return images


def prepare_input(
    img_paths, size, device, raymaps=None, raymap_mask=None, revisit=1, update=True, reset_interval=10000
):
    """
    Prepare input views for inference from a list of image paths.

    Args:
        img_paths (list): List of image file paths.
        img_mask (list of bool): Flags indicating valid images.
        size (int): Target image size.
        raymaps (list, optional): List of ray maps.
        raymap_mask (list, optional): Flags indicating valid ray maps.
        revisit (int): How many times to revisit each view.
        update (bool): Whether to update the state on revisits.

    Returns:
        list: A list of view dictionaries.
    """
    # Import image loader (delayed import needed after adding ckpt path).
    # from models.ttt3r.dust3r.utils.image import load_images

    # images = load_images(img_paths, size=size)
    _images = load_and_resize16(img_paths, resize_to=size, device=device)
    _images = (_images - 0.5) / 0.5  # normalize to [-1, 1]
    images = [
        dict(
            img=_images[i][None],
            true_shape=np.int32(_images[i].shape[-2:]),
            idx=i,
            instance=str(i),
        ) for i in range(_images.shape[0])
    ]
    img_mask = [True] * len(img_paths)
    views = []

    if raymaps is None and raymap_mask is None:
        # Only images are provided.
        for i in range(len(images)):
            view = {
                "img": images[i]["img"],
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ),
                "true_shape": torch.from_numpy(images[i]["true_shape"]),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(True).unsqueeze(0),
                "ray_mask": torch.tensor(False).unsqueeze(0),
                "update": torch.tensor(True).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
    else:
        # Combine images and raymaps.
        num_views = len(images) + len(raymaps)
        assert len(img_mask) == len(raymap_mask) == num_views
        assert sum(img_mask) == len(images) and sum(raymap_mask) == len(raymaps)

        j = 0
        k = 0
        for i in range(num_views):
            view = {
                "img": (
                    images[j]["img"]
                    if img_mask[i]
                    else torch.full_like(images[0]["img"], torch.nan)
                ),
                "ray_map": (
                    raymaps[k]
                    if raymap_mask[i]
                    else torch.full_like(raymaps[0], torch.nan)
                ),
                "true_shape": (
                    torch.from_numpy(images[j]["true_shape"])
                    if img_mask[i]
                    else torch.from_numpy(np.int32([raymaps[k].shape[1:-1][::-1]]))
                ),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ),
                "img_mask": torch.tensor(img_mask[i]).unsqueeze(0),
                "ray_mask": torch.tensor(raymap_mask[i]).unsqueeze(0),
                "update": torch.tensor(img_mask[i]).unsqueeze(0),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0),
            }
            if img_mask[i]:
                j += 1
            if raymap_mask[i]:
                k += 1
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0)
                views.append(overlap_view)
        assert j == len(images) and k == len(raymaps)

    if revisit > 1:
        new_views = []
        for r in range(revisit):
            for i, view in enumerate(views):
                new_view = deepcopy(view)
                new_view["idx"] = r * len(views) + i
                new_view["instance"] = str(r * len(views) + i)
                if r > 0 and not update:
                    new_view["update"] = torch.tensor(False).unsqueeze(0)
                new_views.append(new_view)
        return new_views

    return views


def prepare_output(outputs, revisit=1, use_pose=True):
    """
    Process inference outputs to generate point clouds and camera parameters for visualization.

    Args:
        outputs (dict): Inference outputs.
        revisit (int): Number of revisits per view.
        use_pose (bool): Whether to transform points using camera pose.

    Returns:
        tuple: (points, colors, confidence, camera parameters dictionary)
    """
    from models.ttt3r.dust3r.utils.camera import pose_encoding_to_camera
    from models.ttt3r.dust3r.post_process import estimate_focal_knowing_depth
    from models.ttt3r.dust3r.utils.geometry import geotrf, matrix_cumprod

    # Only keep the outputs corresponding to one full pass.
    valid_length = len(outputs["pred"]) // revisit
    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    # delet overlaps: reset_mask=True outputs["pred"] and outputs["views"]
    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], 0)
    shifted_reset_mask = torch.cat([torch.tensor(False).unsqueeze(0), reset_mask[:-1]], dim=0)

    outputs["pred"] = [
        pred for pred, mask in zip(outputs["pred"], shifted_reset_mask) if not mask]
    outputs["views"] = [
        view for view, mask in zip(outputs["views"], shifted_reset_mask) if not mask]
    reset_mask = reset_mask[~shifted_reset_mask]

    pts3ds_self_ls = [output["pts3d_in_self_view"].cpu() for output in outputs["pred"]]
    pts3ds_other = [output["pts3d_in_other_view"].cpu() for output in outputs["pred"]]
    conf_self = [output["conf_self"].cpu() for output in outputs["pred"]]
    conf_other = [output["conf"].cpu() for output in outputs["pred"]]
    pts3ds_self = torch.cat(pts3ds_self_ls, 0)

    # Recover camera poses.
    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()).cpu()
        for pred in outputs["pred"]
    ]

    if reset_mask.any():
        pr_poses = torch.cat(pr_poses, 0)
        identity = torch.eye(4, device=pr_poses.device)
        reset_poses = torch.where(reset_mask.unsqueeze(-1).unsqueeze(-1), pr_poses, identity)
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat([identity.unsqueeze(0), cumulative_bases[:-1]], dim=0)
        pr_poses = torch.einsum('bij,bjk->bik', shifted_bases, pr_poses)
        # Convert sequence_scale list
        pr_poses = list(pr_poses.unsqueeze(1).unbind(0))

    R_c2w = torch.cat([pr_pose[:, :3, :3] for pr_pose in pr_poses], 0)
    t_c2w = torch.cat([pr_pose[:, :3, 3] for pr_pose in pr_poses], 0)

    if use_pose:
        transformed_pts3ds_other = []
        for pose, pself in zip(pr_poses, pts3ds_self):
            transformed_pts3ds_other.append(geotrf(pose, pself.unsqueeze(0)))
        pts3ds_other = transformed_pts3ds_other
        conf_other = conf_self

    # Estimate focal length based on depth.
    B, H, W, _ = pts3ds_self.shape
    pp = torch.tensor([W // 2, H // 2], device=pts3ds_self.device).float().repeat(B, 1)
    focal = estimate_focal_knowing_depth(pts3ds_self, pp, focal_mode="weiszfeld")

    colors = [
        0.5 * (output["img"].permute(0, 2, 3, 1) + 1.0) for output in outputs["views"]
    ]

    cam_dict = {
        "focal": focal.cpu().numpy(),
        "pp": pp.cpu().numpy(),
        "R": R_c2w.cpu().numpy(),
        "t": t_c2w.cpu().numpy(),
    }

    pts3ds_self_tosave = pts3ds_self  # B, H, W, 3
    depths_tosave = pts3ds_self_tosave[..., 2]
    pts3ds_other_tosave = torch.cat(pts3ds_other)  # B, H, W, 3
    conf_self_tosave = torch.cat(conf_self)  # B, H, W
    conf_other_tosave = torch.cat(conf_other)  # B, H, W
    colors_tosave = torch.cat(
        [
            0.5 * (output["img"].permute(0, 2, 3, 1).cpu() + 1.0)
            for output in outputs["views"]
        ]
    )  # [B, H, W, 3]
    cam2world_tosave = torch.cat(pr_poses)  # B, 4, 4
    intrinsics_tosave = (
        torch.eye(3).unsqueeze(0).repeat(cam2world_tosave.shape[0], 1, 1)
    )  # B, 3, 3
    intrinsics_tosave[:, 0, 0] = focal.detach().cpu()
    intrinsics_tosave[:, 1, 1] = focal.detach().cpu()
    intrinsics_tosave[:, 0, 2] = pp[:, 0]
    intrinsics_tosave[:, 1, 2] = pp[:, 1]

    # if os.path.exists(os.path.join(outdir, "depth")):
    #     shutil.rmtree(os.path.join(outdir, "depth"))
    # if os.path.exists(os.path.join(outdir, "conf")):
    #     shutil.rmtree(os.path.join(outdir, "conf"))
    # if os.path.exists(os.path.join(outdir, "color")):
    #     shutil.rmtree(os.path.join(outdir, "color"))
    # if os.path.exists(os.path.join(outdir, "camera")):
    #     shutil.rmtree(os.path.join(outdir, "camera"))
    # os.makedirs(os.path.join(outdir, "depth"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "conf"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "color"), exist_ok=True)
    # os.makedirs(os.path.join(outdir, "camera"), exist_ok=True)
    # for f_id in range(len(pts3ds_self)):
    #     depth = depths_tosave[f_id].cpu().numpy()
    #     conf = conf_self_tosave[f_id].cpu().numpy()
    #     color = colors_tosave[f_id].cpu().numpy()
    #     c2w = cam2world_tosave[f_id].cpu().numpy()
    #     intrins = intrinsics_tosave[f_id].cpu().numpy()
    #     np.save(os.path.join(outdir, "depth", f"{f_id:06d}.npy"), depth)
    #     np.save(os.path.join(outdir, "conf", f"{f_id:06d}.npy"), conf)
    #     iio.imwrite(
    #         os.path.join(outdir, "color", f"{f_id:06d}.png"),
    #         (color * 255).astype(np.uint8),
    #     )
    #     np.savez(
    #         os.path.join(outdir, "camera", f"{f_id:06d}.npz"),
    #         pose=c2w,
    #         intrinsics=intrins,
    #     )

    # # convert_scene_output_to_glb(outdir, (colors_tosave * 255).to(torch.uint8), pts3ds_other_tosave, conf_other_tosave > 1, focal, cam2world_tosave, as_pointcloud=True)
    return pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave


def infer_videodepth(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    images = prepare_input(
        filelist, size=hydra_cfg.load_img_size, device=hydra_cfg.device
    )
    
    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            outputs, state_args = model(images)
    end = time.time()

    pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave = prepare_output(
        outputs, 1, True
    )
    depth_map = depths_tosave.cpu()  # depth_map (N, H, W)
    depth_conf = conf_other_tosave.cpu()        # depth_conf (N, H, W)
    return  end - start, depth_map, depth_conf



def infer_monodepth(file: str, model: TTT3R, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    images = prepare_input(
        [file], size=hydra_cfg.load_img_size, device=hydra_cfg.device
    )
    

    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            outputs, state_args = model(images)


    pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave = prepare_output(
        outputs, 1, True
    )
    depth_map = depths_tosave.cpu()  # depth_map (N, H, W)
    depth_conf = conf_other_tosave.cpu()        # depth_conf (N, H, W)
    return  depth_map[0].detach()  



def infer_mv_pointclouds(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig, data_size: Tuple[int, int]):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    images = prepare_input(
        filelist, size=hydra_cfg.load_img_size, device=hydra_cfg.device
    )
    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            outputs, state_args = model(images)

    pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave = prepare_output(
        outputs, 1, True
    )
    pts3ds_other_tosave = F.interpolate(
        pts3ds_other_tosave.to(hydra_cfg.device).permute(0, 3, 1, 2), data_size,
        mode="bilinear", align_corners=False, antialias=True
    ).permute(0, 2, 3, 1).cpu().numpy()  # align to gt

    return pts3ds_other_tosave


def infer_cameras_c2w(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    images = prepare_input(
        filelist, size=hydra_cfg.load_img_size, device=hydra_cfg.device
    )
    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            outputs, state_args = model(images)

    pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave = prepare_output(
        outputs, 1, True
    )

    # since we don't eval intrinsics, just return None
    return cam2world_tosave[:, :3, :], None


def infer_cameras_w2c(filelist: List[str], model: TTT3R, hydra_cfg: DictConfig):
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # images = load_and_resize14(filelist, resize_to=hydra_cfg.load_img_size, device=hydra_cfg.device)
    images = prepare_input(
        filelist, size=hydra_cfg.load_img_size, device=hydra_cfg.device
    )
    
    with torch.no_grad():
        with torch.amp.autocast(hydra_cfg.device, dtype=dtype):
            # Predict attributes including cameras, depth maps, and point maps.
            outputs, state_args = model(images)

    pts3ds_other_tosave, colors, depths_tosave, conf_other_tosave, cam2world_tosave = prepare_output(
        outputs, 1, True
    )

    # since we don't eval intrinsics, just return None
    return closed_form_inverse_se3(cam2world_tosave[:, :3, :]), None