import os
import os.path as osp
import hydra
import numpy as np
import json
import logging

from omegaconf import DictConfig
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.messages import set_default_arg
# from utils.debug import setup_debug

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    np.random.seed(hydra_cfg.seed)
    initial_state = np.random.get_state()

    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets
    all_data_info: DictConfig     = hydra_cfg.data
    
    for dataset_idx, dataset_name in enumerate(all_eval_datasets, start=1):
        dataset_logger = logging.getLogger(f"relpose-sampling-{dataset_name}")
        if dataset_name not in all_data_info:
            raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
        dataset_info = all_data_info[dataset_name]
        if osp.exists(dataset_info.seq_id_map):
            dataset_logger.info(f"[{dataset_idx}/{len(all_eval_datasets)}] {dataset_name} has already sampled seq-id-map: {dataset_info.seq_id_map}, skip...")
            continue
        
        dataset_logger.info(f"[{dataset_idx}/{len(all_eval_datasets)}] Creating dataset {dataset_name}...")
        dataset = hydra.utils.instantiate(dataset_info.cfg)

        sample_config: DictConfig = dataset_info.sampling
        seq_id_map = {}
        np.random.set_state(initial_state)

        dataset_logger.info(f"Start sampling ids for {len(dataset.sequence_list)} sequences, Sampling strategy: {sample_config.strategy}")
        dataset_logger.info(f"Sampling strategy full config: {sample_config}")
        for seq_name in tqdm(dataset.sequence_list):
            seq_num_frames = dataset.get_seq_framenum(sequence_name=seq_name)
            if sample_config.strategy == "all":
                num_frames = seq_num_frames
                ids = np.arange(seq_num_frames).tolist()
            elif sample_config.strategy == "random_order":
                num_frames = sample_config.num_frames
                if seq_num_frames < num_frames:
                    dataset_logger.warning(f"[{dataset_name}] sequence {seq_name} has only {seq_num_frames} frames < {num_frames}, skip...")
                    continue
                ids = np.random.choice(seq_num_frames, sample_config.num_frames, replace=False).tolist()
            else:
                raise ValueError(f"Sampling strategy {sample_config.strategy} is not implemented yet.")
            seq_id_map[seq_name] = ids
        
        os.makedirs(osp.dirname(dataset_info.seq_id_map), exist_ok=True)
        with open(dataset_info.seq_id_map, "w") as f:
            json.dump(seq_id_map, f, indent=4)

if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    main()