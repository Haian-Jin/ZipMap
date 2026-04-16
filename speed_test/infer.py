import hydra
import os
import os.path as osp
import torch
import logging
import json
import csv
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from utils.debug import setup_debug
from utils.messages import set_default_arg
import numpy as np
from copy import deepcopy

def prepare_dummy_input_for_cut3r(
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
    # _images = load_and_resize16(img_paths, resize_to=size, device=device)
    h, w = size[0], size[1]  
    _images = torch.zeros((len(img_paths), 3, h, w), device=device)
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
                "img": images[i]["img"].to(device),
                "ray_map": torch.full(
                    (
                        images[i]["img"].shape[0],
                        6,
                        images[i]["img"].shape[-2],
                        images[i]["img"].shape[-1],
                    ),
                    torch.nan,
                ).to(device),
                "true_shape": torch.from_numpy(images[i]["true_shape"]).to(device),
                "idx": i,
                "instance": str(i),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(
                    0
                ).to(device),
                "img_mask": torch.tensor(True).unsqueeze(0).to(device),
                "ray_mask": torch.tensor(False).unsqueeze(0).to(device),
                "update": torch.tensor(True).unsqueeze(0).to(device),
                "reset": torch.tensor((i+1) % reset_interval == 0).unsqueeze(0).to(device),
            }
            views.append(view)
            if (i+1) % reset_interval == 0:
                overlap_view = deepcopy(view)
                overlap_view["reset"] = torch.tensor(False).unsqueeze(0).to(device)
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


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    logger = logging.getLogger("SpeedTest")

    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/videodepth.yaml
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    # Store results for CSV output
    results = []

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]
        
        # 0.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        model.eval()
    
        logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")

        Test_interation = 2+10  # total 12 runs, skip the first 2 for warm-up, report the average of the last 10 runs
        for input_img_num in [5, 10, 25, 50, 100, 200, 300, 400, 500, 750]:
            torch.cuda.synchronize()
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            run_time_list = []
            if model_keyname == 'cut3r' or model_keyname == 'ttt3r':
                input_image = prepare_dummy_input_for_cut3r([0]*input_img_num, [384, 512], device=hydra_cfg.device)
            else:
                input_image = torch.zeros((1, input_img_num, 3, 392,518)).to(hydra_cfg.device)

            for _ in range(Test_interation):
                torch.cuda.synchronize()
                start_time.record()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # outputs = model(input_image)
                    if model_keyname == 'cut3r' or model_keyname == 'ttt3r':
                        outputs, temp = model(input_image)
                    else:
                        outputs = model(input_image)
                end_time.record()
                end_time.synchronize()
                elapsed_time = start_time.elapsed_time(end_time)
                run_time_list.append(elapsed_time)
            print(model_keyname, run_time_list)
            
            # skip the first two runs for warm-up
            run_time_list = run_time_list[2:]
            mean_time = sum(run_time_list) / len(run_time_list)
            print(f"Mean inference time for {input_img_num} images: {mean_time:.2f} ms")
            logger.info(f"[{idx_model}/{len(all_eval_models)}] Inference time for {input_img_num} images: {elapsed_time:.2f} ms")

            # Store result
            results.append({
                'model': model_keyname,
                'input_img_num': input_img_num,
                'mean_time_ms': mean_time
            })

        del model
        torch.cuda.empty_cache()

    # Save results to CSV with models as rows and input_img_num as columns
    csv_path = os.path.join(hydra_cfg.get('output_dir', './'), 'speed_test_results.csv')
    os.makedirs(os.path.dirname(csv_path) if os.path.dirname(csv_path) else '.', exist_ok=True)

    # Pivot the results: organize by model and input_img_num
    model_results = {}
    all_input_nums = set()

    for result in results:
        model = result['model']
        input_num = result['input_img_num']
        mean_time = result['mean_time_ms']

        if model not in model_results:
            model_results[model] = {}
        model_results[model][input_num] = mean_time
        all_input_nums.add(input_num)

    # Sort input numbers for consistent column order
    sorted_input_nums = sorted(all_input_nums)

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model'] + sorted_input_nums
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for model in sorted(model_results.keys()):
            row = {'model': model}
            for input_num in sorted_input_nums:
                row[input_num] = f"{model_results[model].get(input_num, '')/1000:.3f}"
            writer.writerow(row)

    logger.info(f"Results saved to {csv_path}")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    set_default_arg("evaluation", "speed_test")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()