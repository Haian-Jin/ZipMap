import os
import json
import torch
import torch.distributed as dist
import numpy as np
import open3d as o3d
import os.path as osp
import hydra
import logging
import pandas as pd

from omegaconf import DictConfig
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from mv_recon.utils import umeyama, accuracy, completion
from utils.messages import set_default_arg, write_csv, gather_csv_and_write, make_csvsdir_and_remove_history_csvs
from utils.vis_utils import save_image_grid_auto


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    if not torch.cuda.is_available() or hydra_cfg.device != "cuda":
        raise EnvironmentError("Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage")
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, world_size={dist.get_world_size()}.")

    all_eval_models: DictConfig   = hydra_cfg.eval_models    # see configs/evaluation/mv_recon.yaml
    all_eval_datasets: DictConfig = hydra_cfg.eval_datasets  # see configs/evaluation/mv_recon.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model, decide the model name (to save)
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]

        # 0.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        model_logger = logging.getLogger(f"mv_recon-eval-{model_keyname}-rank{rank}")
        model_logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")
        
        # 0.3 route the correct infer function for the model
        infer_func_cfg = model_info.get(
            "infer_mv_pointclouds",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_mv_pointclouds',
                '_partial_': True,
            })
        )
        infer_mv_pointclouds = hydra.utils.instantiate(infer_func_cfg)

        all_datasets = {}
        all_seq_ids = []
        for dataset_name in all_eval_datasets:
            # 1.1 look up dataset config from configs/data, decide the dataset name, and load the dataset
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
            dataset_info = all_data_info[dataset_name]
            all_datasets[dataset_name] = hydra.utils.instantiate(dataset_info.cfg)

            # 1.2 load pre-sampled seq-id-map & remove old sequence csv file
            with open(dataset_info.seq_id_map, "r") as f:
                seq_id_map = json.load(f)
            all_seq_ids.extend([(dataset_name, seq_name, ids) for seq_name, ids in seq_id_map.items()])

            # 1.3 ready for output directory
            output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
            if rank == 0:
                make_csvsdir_and_remove_history_csvs(
                    input_root=osp.join(output_root, "_seq_metrics"),
                    seqs_csv_file=osp.join(output_root, "_seq_metrics.csv")
                )
        
        dist.barrier()  # make sure all history "_all_samples.csv" are removed

        model_logger.info(f"Start evaluating point map...")
        output_root = osp.join(hydra_cfg.output_dir, model_keyname)
        seq_ids_this_rank = all_seq_ids[rank::dist.get_world_size()]
        tbar = tqdm(
            seq_ids_this_rank,
            desc=f"[Rank {rank}] {model_keyname}"
        )
        for dataset_name, seq_name, ids in tbar:
            save_dir = osp.join(output_root, dataset_name)

            # 2. load data, choose specific ids of a sequence
            data = all_datasets[dataset_name].get_data(sequence_name=seq_name, ids=ids)
            filelist: list         = data['image_paths']  # [str] * N
            images: torch.Tensor   = data['images']       # (N, 3, H, W)
            gt_pts: np.ndarray     = data['pointclouds']  # (N, H, W, 3)
            valid_mask: np.ndarray = data['valid_mask']   # (N, H, W)

            # 3. real inference, predicted pointcloud aligned to ground truth (data_h, data_w)
            data_h, data_w         = images.shape[-2:]
            pred_pts: np.ndarray   = infer_mv_pointclouds(filelist, model, hydra_cfg, (data_h, data_w))  # (N, H, W, 3)
            assert pred_pts.shape == gt_pts.shape, f"Predicted points shape {pred_pts.shape} does not match ground truth shape {gt_pts.shape}."

            # 4. save input images
            seq_name = seq_name.replace("/", "-")
            save_image_grid_auto(images, osp.join(save_dir, f"{seq_name}.png"))
            colors = images.permute(0, 2, 3, 1)[valid_mask].cpu().numpy().reshape(-1, 3)

            # 5. coarse align
            c, R, t = umeyama(pred_pts[valid_mask].T, gt_pts[valid_mask].T)
            pred_pts = c * np.einsum('nhwj, ij -> nhwi', pred_pts, R) + t.T

            # 6. filter invalid points
            pred_pts = pred_pts[valid_mask].reshape(-1, 3)
            gt_pts = gt_pts[valid_mask].reshape(-1, 3)

            # 7. save predicted & ground truth point clouds
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pred_pts)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(save_dir, f"{seq_name}-pred.ply"), pcd)

            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(gt_pts)
            pcd_gt.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(osp.join(save_dir, f"{seq_name}-gt.ply"), pcd_gt)

            # 8. ICP align refinement
            if "DTU" in dataset_name:
                threshold = 100
            else:
                threshold = 0.1

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd,
                pcd_gt,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )

            transformation = reg_p2p.transformation
            pcd = pcd.transform(transformation)
            
            # 9. estimate normals
            pcd.estimate_normals()
            pcd_gt.estimate_normals()
            pred_normal = np.asarray(pcd.normals)
            gt_normal = np.asarray(pcd_gt.normals)

            # o3d.io.write_point_cloud(
            #     os.path.join(
            #         save_path, f"{seq.replace('/', '_')}-mask-icp.ply"
            #     ),
            #     pcd,
            # )

            # 10. compute metrics
            acc,  acc_med,  nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
            comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)
            tbar.set_postfix_str(f"{dataset_name}-{seq_name}")  # too many metrics, so not to show

            # 11. save metrics to csv
            write_csv(osp.join(save_dir, "_seq_metrics", f"{seq_name}.csv"), {
                "seq":       seq_name,
                "Acc-mean":  acc,
                "Acc-med":   acc_med,
                "Comp-mean": comp,
                "Comp-med":  comp_med,
                'NC-mean':   (nc1 + nc2) / 2,
                'NC-med':    (nc1_med + nc2_med) / 2,
                "NC1-mean":  nc1,
                "NC1-med":   nc1_med,
                "NC2-mean":  nc2,
                "NC2-med":   nc2_med,
            })

            # release cuda memory
            torch.cuda.empty_cache()

        dist.barrier()  # make sure all samples are processed
        if rank == 0:
            for dataset_name in all_eval_datasets:
                save_dir = osp.join(output_root, dataset_name)
                df = gather_csv_and_write(
                    input_root=osp.join(save_dir, "_seq_metrics"),
                    output_file=osp.join(save_dir, "_seq_metrics.csv")
                )
                
                mean_values = df.mean(numeric_only=True)
                metric_dict = { "model": model_keyname }
                for metric in ["Acc-mean", "Acc-med", "Comp-mean", "Comp-med", "NC-mean", "NC-med", "NC1-mean", "NC1-med", "NC2-mean", "NC2-med"]:
                    metric_dict[metric] = mean_values[metric].item()

                statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
                if getattr(hydra_cfg, "save_suffix", None) is not None:
                    statistics_file += f"-{hydra_cfg.save_suffix}"
                statistics_file += ".csv"
                write_csv(statistics_file, metric_dict)
        
        del model
        torch.cuda.empty_cache()
        model_logger.info(f"Finished evaluating {model_keyname} on all datasets.")

        # Make sure all processes have finished
        dist.barrier()
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    set_default_arg("evaluation", "mv_recon")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()