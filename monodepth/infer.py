import hydra
import os
import os.path as osp
import numpy as np
import cv2
import logging
import torch

from tqdm import tqdm
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.files import list_imgs_a_sequence, get_all_sequences
# from utils.debug import setup_debug
from utils.messages import set_default_arg


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    logger = logging.getLogger("monodepth-infer")

    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/monodepth.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data  
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]
        
        # 0.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")

        # 0.3 look up infer_monodepth function
        infer_func_cfg = model_info.get(
            "infer_monodepth",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_monodepth',
                '_partial_': True,
            })
        )
        infer_monodepth = hydra.utils.instantiate(infer_func_cfg)

        model_logger = logging.getLogger(f"monodepth-infer-{model_keyname}")
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            # 1. look up dataset config from configs/data
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            dataset_info = all_data_info[dataset_name]

            # 2. get the sequence list
            if dataset_info.type == "video":
                # most of the datasets have many sequences of video
                seq_list = get_all_sequences(dataset_info)
            elif dataset_info.type == "mono":
                # some datasets (like nyu-v2) have only a set of images, only for monodepth
                seq_list = [None]
            else:
                raise ValueError(f"Unknown dataset type: {dataset_info.type}")

            # 3. infer for each sequence
            model = model.eval()
            output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
            model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering monodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")
            for seq_idx, seq in enumerate(seq_list):
                # 3.1 list the images in the sequence
                filelist = list_imgs_a_sequence(dataset_info, seq)
                save_dir = osp.join(output_root, seq) if seq is not None else output_root
                os.makedirs(save_dir, exist_ok=True)
                model_logger.info(f"[{seq_idx}/{len(seq_list)}] Processing {len(filelist)} images to {osp.relpath(save_dir, hydra_cfg.work_dir)}...")

                # 3.2 infer for each image
                for file in tqdm(filelist):
                    # 3.2.1 skip if the file already exists
                    npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.npy'))
                    png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.png'))
                    if not hydra_cfg.overwrite and (osp.exists(npy_save_path) and osp.exists(png_save_path)):
                        continue

                    # 3.2.2 infer the depth map
                    depth_map = infer_monodepth(file, model, hydra_cfg)

                    # 3.2.3 save the depth map to the save_dir as npy
                    if isinstance(depth_map, torch.Tensor):
                        depth_map = depth_map.cpu().numpy()
                    elif not isinstance(depth_map, np.ndarray):
                        raise ValueError(f"Unknown depth map type: {type(depth_map)}")
                    np.save(npy_save_path, depth_map)

                    # 3.2.4 also save the png
                    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
                    depth_map = (depth_map * 255).astype(np.uint8)
                    cv2.imwrite(png_save_path, depth_map)
            # for each dataset
            model_logger.info(f"Monodepth inference for dataset {dataset_name} finished!")
        # for each model
        del model
        torch.cuda.empty_cache()
        logger.info(f"Monodepth inference for model {model_keyname} finished!")

if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()
