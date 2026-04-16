import hydra
import os
import os.path as osp
import numpy as np
import cv2
import logging
import torch
import torch.distributed as dist

from tqdm import tqdm
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    if not torch.cuda.is_available() or hydra_cfg.device != "cuda":
        raise EnvironmentError("Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/monodepth.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data  
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 1.1 look up model config from configs/model
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]
        
        # 1.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        model_logger = logging.getLogger(f"monodepth-infer-{model_keyname}-rank{rank}")
        model_logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")

        # 1.3 look up infer_monodepth function
        infer_func_cfg = model_info.get(
            "infer_monodepth",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_monodepth',
                '_partial_': True,
            })
        )
        infer_monodepth = hydra.utils.instantiate(infer_func_cfg)
        
        # 2. gather all files together
        all_file_list = []
        for dataset_name in all_eval_datasets:
            # 2.1. look up dataset config from configs/data
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            dataset_info = all_data_info[dataset_name]

            # 2.2 get the sequence list for each dataset
            if dataset_info.type == "video":
                # most of the datasets have many sequences of video
                seq_list = get_all_sequences(dataset_info)
            elif dataset_info.type == "mono":
                # some datasets (like nyu-v2) have only a set of images, only for monodepth
                seq_list = [None]
            else:
                raise ValueError(f"Unknown dataset type: {dataset_info.type}")
            
            # 2.3 add the dataset and sequence info to the all_seq_list
            for seq in seq_list:
                filelist = list_imgs_a_sequence(dataset_info, seq)
                all_file_list.extend([(dataset_name, seq, file) for file in filelist])

        # 3. infer for each sequence
        model = model.eval()
        output_root = osp.join(hydra_cfg.output_dir, model_keyname)
        model_logger.info(f"Start infering monodepth on rank {rank}..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")
        
        file_list_this_rank = all_file_list[rank::dist.get_world_size()]
        tbar = tqdm(
            file_list_this_rank,
            desc=f"[Rank {rank}] {model_keyname}"
        )
        for dataset_name, seq, file in tbar:
            # 3.1 list the images in the sequence
            save_dir = osp.join(output_root, dataset_name, seq) if seq is not None else osp.join(output_root, dataset_name)
            os.makedirs(save_dir, exist_ok=True)

            # 3.2.1 skip if the file already exists
            npy_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.npy'))
            png_save_path = osp.join(save_dir, file.split('/')[-1].replace('.png', 'depth.png'))
            if not hydra_cfg.overwrite and (osp.exists(npy_save_path) and osp.exists(png_save_path)):
                continue

            # 3.2.2 infer the depth map, torch.Tensor or np.ndarray of shape (H, W)
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

            tbar.set_postfix_str(f"{dataset_name}{f'-{seq}' if seq is not None else ''}-{osp.basename(file)}")

        # for each model
        del model
        torch.cuda.empty_cache()
        model_logger.info(f"Monodepth inference for model {model_keyname} finished!")

        # Make sure all processes have finished
        dist.barrier()
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()
