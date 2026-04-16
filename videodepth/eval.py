import os
import os.path as osp
import hydra
import numpy as np
import cv2
import json
import logging
import glob
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.depth import EVAL_DEPTH_METADATA, depth_evaluation
from utils.files import get_all_sequences, list_depths_a_sequence
# from utils.debug import setup_debug
from utils.messages import set_default_arg, write_csv

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    logger = logging.getLogger("videodepth-eval")
    
    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/videodepth.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/videodepth.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0. decide the model name (to save)
        model_logger = logging.getLogger(f"monodepth-eval-{model_keyname}")
        model_logger.info(f"[{idx_model}/{len(all_eval_models)}] Start evaluating {model_keyname} on {len(all_eval_datasets)} datasets...")

        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            # 1. look up dataset config from configs/data, decide the dataset name
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
            dataset_info = all_data_info[dataset_name]
            max_n_frames = dataset_info.get("num_frames", None)

            # 2. get gt and pred depth pathes
            output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
            if dataset_info.type == "video":
                # most of the datasets have many sequences of video
                seq_list = get_all_sequences(dataset_info)
                gt_paths = {
                    seq: list_depths_a_sequence(dataset_info, seq)
                    for seq in seq_list
                }
                if max_n_frames is not None:
                    for seq in seq_list:
                        gt_paths[seq] = gt_paths[seq][:max_n_frames]
                pred_paths = {
                    seq: sorted(glob.glob(f"{output_root}/{seq}/*.npy"))
                    for seq in seq_list
                }
                for seq in seq_list:
                    if len(pred_paths[seq]) != len(gt_paths[seq]):
                        raise ValueError(f"Number of prediction and ground truth depth maps are not equal: {len(pred_paths[seq])} vs {len(gt_paths[seq])}")
            else:
                raise ValueError(f"Unknown dataset type: {dataset_info.type}")
            
            # 3. get depth read function and evaluation kwargs
            videodepth_metadata = EVAL_DEPTH_METADATA.get(dataset_name, None)
            if videodepth_metadata is None:
                videodepth_metadata = EVAL_DEPTH_METADATA.get(dataset_name.split('-')[0], None)
                if videodepth_metadata is None:
                    raise ValueError(f"Dataset {dataset_name} doesn't have video depth metadata")
            depth_read_func = videodepth_metadata["depth_read_func"]
            depth_evaluation_kwargs = videodepth_metadata["depth_evaluation_kwargs"]
            if hydra_cfg.align == "scale&shift":
                depth_evaluation_kwargs["align_with_lad2"] = True
            elif hydra_cfg.align == "scale":
                depth_evaluation_kwargs["align_with_scale"] = True
            elif hydra_cfg.align == "metric":
                depth_evaluation_kwargs["metric_scale"] = True
            else:
                raise ValueError(f"Unknown alignment method: {hydra_cfg.align}")

            gathered_depth_metrics = []
            all_fps = []
            total_time_per_seq = []
            infer_time_per_seq = []

            model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Start evaluating dataset: {dataset_name}, {len(pred_paths)} sequences in total")
            for idx_seq, seq in enumerate(pred_paths, start=1):
                # 3.1 get pred & gt depths for each seq
                seq_pd_paths = pred_paths[seq]
                seq_gt_paths = gt_paths[seq]
                
                model_logger.info(f"[{idx_seq}/{len(pred_paths)}] Evaluating {seq} in {dataset_name}, {len(seq_pd_paths)} images in total")
                
                # 3.2 pack the pred & gt depths
                gt_depth = np.stack([depth_read_func(gt_path) for gt_path in seq_gt_paths], axis=0)
                pr_depth = np.stack(
                    [
                        cv2.resize(
                            np.load(pd_path),
                            (gt_depth.shape[2], gt_depth.shape[1]),
                            interpolation=cv2.INTER_CUBIC,
                        )
                        for pd_path in seq_pd_paths
                    ],
                    axis=0,
                )
                
                # 3.3 evaluate depth
                # for depth eval, set align_with_lad2=False to use median alignment; set align_with_lad2=True to use scale&shift alignment
                depth_results, error_map, depth_predict, depth_gt = depth_evaluation(pr_depth, gt_depth, **depth_evaluation_kwargs)
                gathered_depth_metrics.append(depth_results)

                # 3.4 calculate fps
                with open(osp.join(output_root, seq, f"_time.json"), "r") as f:
                    timing_data = json.load(f)
                timing = timing_data["time"]
                if isinstance(timing, list):
                    total_time = sum(timing)
                    total_time_per_seq.append(total_time)
                    infer_time_per_seq.append(timing[0])
                elif isinstance(timing, float):
                    total_time = timing
                    total_time_per_seq.append(timing)
                    infer_time_per_seq.append(timing)
                else:
                    raise ValueError(f"Unknown timing type: {type(timing)}")
                all_fps.append(timing_data["frames"] / total_time)

                # 3.5 calculate metrics for this sequence
                this_seq_metrics = depth_results
                this_seq_metrics["fps"] = all_fps[-1]
                this_seq_metrics["total_time"] = total_time_per_seq[-1]
                this_seq_metrics["infer_time"] = infer_time_per_seq[-1]
                model_logger.info(f"{dataset_name} - seq {seq} - videodepth metrics: {depth_results}")

                # seq_len = gt_depth.shape[0]
                # error_map = error_map.reshape(seq_len, -1, error_map.shape[-1]).cpu()
                # error_map_colored = colorize(error_map, range=(error_map.min(), error_map.max()), append_cbar=True)
                # ImageSequenceClip([x for x in (error_map_colored.numpy()*255).astype(np.uint8)], fps=10).write_videofile(f'{args.output_dir}/errormap_{key}_{args.align}.mp4', fps=10)
            
            # 4. save evaluation metrics to csv
            total_average_metrics = {
                key: np.average(
                    [metrics[key] for metrics in gathered_depth_metrics],
                    weights=[
                        metrics["valid_pixels"] for metrics in gathered_depth_metrics
                    ],
                )
                for key in gathered_depth_metrics[0].keys()
                if key != "valid_pixels"
            }
            total_average_metrics["fps"] = np.average(all_fps).item()
            total_average_metrics["total_time"] = np.average(total_time_per_seq).item()
            total_average_metrics["infer_time"] = np.average(infer_time_per_seq).item()
            model_logger.info(f"{dataset_name} - Average Videodepth Metrics: {total_average_metrics}")
            
            statistics_data = {"model": model_keyname, **total_average_metrics}
            statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric-{hydra_cfg.align}")  # + ".csv"
            if getattr(hydra_cfg, "save_suffix", None) is not None:
                statistics_file += f"-{hydra_cfg.save_suffix}"
            statistics_file += ".csv"
            write_csv(statistics_file, statistics_data)

if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    main()
