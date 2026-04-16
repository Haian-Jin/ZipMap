import os
import os.path as osp
import logging
import numpy as np
import torch
import torch.distributed as dist
import hydra
import pandas as pd

from tqdm import tqdm
from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.files import list_imgs_a_sequence, get_all_sequences
from utils.messages import set_default_arg, write_csv, save_list_of_matrices, gather_csv_and_write, make_csvsdir_and_remove_history_csvs
from relpose.evo_utils import load_traj, eval_metrics, plot_trajectory, get_tum_poses, save_tum_poses


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    if not torch.cuda.is_available() or hydra_cfg.device != "cuda":
        raise EnvironmentError("Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    all_eval_models: DictConfig   = hydra_cfg.eval_models    # see configs/evaluation/relpose-distance.yaml
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-distance.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model, decide the model name (to save)
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]
        
        # 0.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        model_logger = logging.getLogger(f"relpose-dist-{model_keyname}-rank{rank}")
        model_logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")

        # 0.3 route the correct infer function for the model
        infer_func_cfg = model_info.get(
            "infer_cameras_c2w",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_cameras_c2w',
                '_partial_': True,
            })
        )
        infer_cameras_c2w = hydra.utils.instantiate(infer_func_cfg)

        all_seq_list = []
        output_root = osp.join(hydra_cfg.output_dir, model_keyname)
        for dataset_name in all_eval_datasets:
            # 1. look up dataset config from configs/data, decide the dataset name
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset: {dataset_name}")
            dataset_info = all_data_info[dataset_name]

            # 2. get the sequence list
            seq_list = get_all_sequences(dataset_info)
            all_seq_list.extend([(dataset_name, seq) for seq in seq_list])

            save_dir = osp.join(output_root, dataset_name)
            if rank == 0:
                make_csvsdir_and_remove_history_csvs(
                    input_root=osp.join(save_dir, "_seq_metrics"),
                    seqs_csv_file=osp.join(save_dir, "_seq_metrics.csv")
                )

        # 3. infer for each sequence
        model = model.eval()
        model_logger.info(f"Start infering relpose(c2w) on dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

        seq_list_this_rank = all_seq_list[rank::dist.get_world_size()]
        tbar = tqdm(
            seq_list_this_rank,
            desc=f"[Rank {rank}] {model_keyname}"
        )
        for dataset_name, seq in tbar:
            try:
                dataset_info = all_data_info[dataset_name]
                save_dir = osp.join(output_root, dataset_name)

                # 4.1 list all images of this sequence
                filelist = list_imgs_a_sequence(dataset_info, seq)
                filelist = filelist[:: hydra_cfg.pose_eval_stride]

                # 4.2 real inference
                # pr_poses: c2w poses, (N, 3, 4), in torch
                # pr_intrs: focals + pps, (N, 3, 3), in numpy
                pr_poses, pr_intrs = infer_cameras_c2w(filelist, model, hydra_cfg)
                pred_traj = get_tum_poses(pr_poses)

                # 4.3 save predicted poses & intrinsics
                seq_save_dir = osp.join(save_dir, seq)
                os.makedirs(seq_save_dir, exist_ok=True)
                # save predicted poses
                save_tum_poses(pred_traj, osp.join(save_dir, seq, "pred_traj.txt"), verbose=hydra_cfg.verbose)
                np.save(osp.join(seq_save_dir, "pred_poses.npy"), pr_poses)
                save_list_of_matrices(pr_poses.numpy().tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))
                # save predicted intrinsics (if available)
                if pr_intrs is not None:
                    np.save(osp.join(seq_save_dir, "pred_intrinsics.npy"), pr_intrs)
                    save_list_of_matrices(pr_intrs.tolist(), osp.join(seq_save_dir, "pred_intrinsics.json"))

                # 4.4 read ground truth trajectory
                gt_traj = load_traj(
                    gt_traj_file = dataset_info.anno.path.format(seq=seq),
                    traj_format  = dataset_info.anno.format,
                    stride       = hydra_cfg.pose_eval_stride,
                )

                # 4.5 evaluate predicted trajectory with ground truth trajectory, plot the trajectory
                if gt_traj is not None:
                    ate, rpe_trans, rpe_rot = eval_metrics(
                        pred_traj, gt_traj,
                        seq      = seq,
                        filename = osp.join(save_dir, seq, "eval_metric.txt"),
                        verbose  = hydra_cfg.verbose,
                    )
                    plot_trajectory(pred_traj, gt_traj, title=seq, filename=osp.join(save_dir, seq, "vis.png"), verbose=hydra_cfg.verbose)
                else:
                    raise ValueError(f"Ground truth trajectory not found for sequence {seq} in dataset {dataset_name}.")

                # 4.6 save sequence metrics to csv
                seq_metrics = {
                    "model": model_keyname,
                    "dataset": dataset_name,
                    "seq": seq,
                    "ATE": ate,
                    "RPE trans": rpe_trans,
                    "RPE rot": rpe_rot,
                }
                seq = seq.replace("/", "-")  # replace '/' with '-' for file name
                write_csv(osp.join(save_dir, "_seq_metrics", f"{seq}.csv"), seq_metrics)
                # write_csv(osp.join(save_dir, "seq_metrics.csv"), seq_metrics)
                # results.append((seq, ate, rpe_trans, rpe_rot))

                # 4.7. update metric for a sequence to tqdm bar
                tbar.set_postfix_str(f"Seq {seq} ATE: {ate:5.2f} | RPE-trans: {rpe_trans:5.2f} | RPE-rot: {rpe_rot:5.2f}")

            except Exception as e:
                if "out of memory" in str(e):
                    # Handle OOM
                    torch.cuda.empty_cache()  # Clear the CUDA memory
                    with open(osp.join(save_dir, "error_log.txt"), "a") as f:
                        f.write(
                            f"OOM error in sequence {seq}, skipping this sequence.\n"
                        )
                    print(f"OOM error in sequence {seq}, skipping...")
                elif "Degenerate covariance rank" in str(
                    e
                ) or "Eigenvalues did not converge" in str(e):
                    # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                    with open(osp.join(save_dir, "error_log.txt"), "a") as f:
                        f.write(f"Exception in sequence {seq}: {str(e)}\n")
                    print(f"Traj evaluation error in sequence {seq}, skipping.")
                else:
                    raise e  # Rethrow if it's not an expected exception

        dist.barrier()  # Ensure all processes finish before proceeding
        if rank == 0:
            for dataset_name in all_eval_datasets:
                save_dir = osp.join(output_root, dataset_name)
                df = gather_csv_and_write(
                    input_root=osp.join(save_dir, "_seq_metrics"),
                    output_file=osp.join(save_dir, "_seq_metrics.csv")
                )
                
                mean_values = df.mean(numeric_only=True)
                metric_dict = { "model": model_keyname }
                for metric in ["ATE", "RPE trans", "RPE rot"]:
                    metric_dict[metric] = mean_values[metric].item()

                statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
                if getattr(hydra_cfg, "save_suffix", None) is not None:
                    statistics_file += f"-{hydra_cfg.save_suffix}"
                statistics_file += ".csv"
                write_csv(statistics_file, metric_dict)

                metric_dict.pop("model")  # Remove model name for logging
                model_logger.info(f"{dataset_name} - Average pose estimation metrics: {metric_dict}")
        
        del model
        torch.cuda.empty_cache()
        # Make sure all processes have finished
        dist.barrier()
    
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-distance")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    main()