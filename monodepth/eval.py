import os
import os.path as osp
import numpy as np
import cv2
import logging
import hydra
import glob

from tqdm import tqdm
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.depth import EVAL_DEPTH_METADATA, depth_evaluation
# from utils.debug import setup_debug
from utils.files import get_all_sequences, list_depths_a_sequence
from utils.messages import set_default_arg, write_csv

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)

    logger = logging.getLogger("monodepth-eval")

    if hydra_cfg.invariant == "median-scale":
        align_with_scale = True
    elif hydra_cfg.invariant == "scale":
        align_with_scale = False
    else:
        raise NotImplementedError(f"Unknown invariant {hydra_cfg.invariant}")
    logger.info(f"Evaluating with invariant depthmap type: {hydra_cfg.invariant}")

    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/monodepth.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/monodepth.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0. decide the model name (to save)
        model_logger = logging.getLogger(f"monodepth-eval-{model_keyname}")
        model_logger.info(f"[{idx_model}/{len(all_eval_models)}] Start evaluating {model_keyname} on {len(all_eval_datasets)} datasets...")
        
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            # 1. look up dataset config from configs/data, decide the dataset name
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            dataset_info = all_data_info[dataset_name]

            # 2. get gt and pred depth pathes
            output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
            if dataset_info.type == "video":
                # most of the datasets have many sequences of video
                seq_list = get_all_sequences(dataset_info)
                gt_paths = {
                    seq: list_depths_a_sequence(dataset_info, seq)
                    for seq in seq_list
                }
                pred_paths = {
                    seq: sorted(glob.glob(f"{output_root}/{seq}/*.npy"))
                    for seq in seq_list
                }
            elif dataset_info.type == "mono":
                seq_list = [dataset_name]
                # some datasets (like nyu-v2) have only a set of images, only for monodepth
                gt_paths = {dataset_name: list_depths_a_sequence(dataset_info, seq=None)}
                pred_paths = {dataset_name: sorted(glob.glob(f"{output_root}/*.npy"))}
            else:
                raise ValueError(f"Unknown dataset type: {dataset_info.type}")
            
            # 3. get depth read function and evaluation kwargs
            mono_metadata = EVAL_DEPTH_METADATA.get(dataset_name, None)
            if mono_metadata is None:
                raise ValueError(f"Dataset {dataset_name} doesn't have monodepth metadata")
            depth_read_func = mono_metadata["depth_read_func"]
            depth_evaluation_kwargs = mono_metadata["depth_evaluation_kwargs"]
            depth_evaluation_kwargs["align_with_scale"] = align_with_scale

            model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Start evaluating dataset: {dataset_name}, {len(seq_list)} sequences in total")
            total_gathered_depth_metrics = []
            for idx_seq, seq in enumerate(seq_list, start=1):
                # 4. check whether the inference result is complete
                seq_gt_paths = gt_paths[seq]    
                seq_pred_paths = pred_paths[seq]
                if len(seq_gt_paths) != len(seq_pred_paths):
                    raise ValueError(f"Number of prediction and ground truth depth maps are not equal: {len(seq_pred_paths)} vs {len(seq_gt_paths)}, output_root: {output_root}, seq: {seq}")
                
                gathered_depth_metrics = []
                for idx in tqdm(range(len(seq_gt_paths))):
                    # 5. read gt and pred depth, resize pred depth to gt depth size
                    pred_depth = np.load(seq_pred_paths[idx])
                    gt_depth = depth_read_func(seq_gt_paths[idx])
                    pred_depth = cv2.resize(
                        pred_depth,
                        (gt_depth.shape[1], gt_depth.shape[0]),
                        interpolation=cv2.INTER_CUBIC,
                    )

                    # 5. evaluate depth
                    depth_results, error_map, depth_predict, depth_gt = depth_evaluation(
                        pred_depth, gt_depth, **depth_evaluation_kwargs
                    )
                    gathered_depth_metrics.append(depth_results)

                # 6. calculate average evaluation metrics for each sequence
                average_metrics = {
                    key: np.average(
                        [metrics[key] for metrics in gathered_depth_metrics],
                        weights=[metrics["valid_pixels"] for metrics in gathered_depth_metrics],
                    ).item()
                    for key in gathered_depth_metrics[0].keys()
                    if key != "valid_pixels"
                }
                model_logger.info(f"{dataset_name} - {seq}({idx_seq}/{len(seq_list)}) - Sequence Monodepth Metrics: {average_metrics}")
                total_gathered_depth_metrics.extend(gathered_depth_metrics)

            # 7. calculate average evaluation metrics for total dataset, save to csv
            average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in total_gathered_depth_metrics],
                    weights=[metrics["valid_pixels"] for metrics in total_gathered_depth_metrics],
                ).item()
                for key in total_gathered_depth_metrics[0].keys()
                if key != "valid_pixels"
            }
            model_logger.info(f"{dataset_name} - Average Monodepth Metrics: {average_metrics}")
            statistics_data = {"model": model_keyname, **average_metrics}
            statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
            if getattr(hydra_cfg, "save_suffix", None) is not None:
                statistics_file += f"-{hydra_cfg.save_suffix}"
            statistics_file += ".csv"
            write_csv(statistics_file, statistics_data)

if __name__ == "__main__":
    set_default_arg("evaluation", "monodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    main()
