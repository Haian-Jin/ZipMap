
import os
import os.path as osp
import numpy as np
import torch
import hydra
import logging
import json

from omegaconf import DictConfig, ListConfig
from tqdm import tqdm

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# from utils.debug import setup_debug
from utils.messages import set_default_arg, write_csv
from relpose.metric import se3_to_relative_pose_error, calculate_auc_np

@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    # OmegaConf.set_struct(hydra_cfg, False)

    logger = logging.getLogger("relpose-angle")

    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/relpose-angular.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/relpose-angular.yaml
    all_data_info: DictConfig     = hydra_cfg.data           # see configs/data
    all_model_info: DictConfig    = hydra_cfg.model          # see configs/model

    for idx_model, model_keyname in enumerate(all_eval_models, start=1):
        # 0.1 look up model config from configs/model, decide the model name (to save)
        if model_keyname not in all_model_info:
            raise ValueError(f"Unknown model in global data information: {model_keyname}")
        model_info = all_model_info[model_keyname]

        # 0.2 load the model
        model = hydra.utils.instantiate(model_info.cfg).to(hydra_cfg.device)
        logger.info(f"[{idx_model}/{len(all_eval_models)}] Loaded Model {model_keyname} from {model_info.cfg.pretrained_model_name_or_path if hasattr(model_info.cfg, 'pretrained_model_name_or_path') else '???'}")

        # 0.3 route the correct infer function for the model
        # output_root = osp.join(hydra_cfg.log.output_dir, model_name)
        infer_func_cfg = model_info.get(
            "infer_cameras_w2c",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_cameras_w2c',
                '_partial_': True,
            })
        )
        infer_cameras_w2c = hydra.utils.instantiate(infer_func_cfg)
        
        model_logger = logging.getLogger(f"relpose-angle-{model_keyname}")
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            # 1. look up dataset config from configs/data, decide the dataset name
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
            dataset_info = all_data_info[dataset_name]
            dataset = hydra.utils.instantiate(dataset_info.cfg)

            # 2. ready to read, and look up sampled ids from sequence name
            model.eval()
            sample_config: DictConfig = dataset_info.sampling
            model_logger.info(f"Sampling strategy: {sample_config.strategy}")
            with open(dataset_info.seq_id_map, "r") as f:
                seq_id_map = json.load(f)

            # 3. prepare for metrics
            rError = []
            tError = []
            metric_dict: dict = {}
            model_logger.info(f"Evaluating {dataset_name} with {model_keyname}...")
            tbar = tqdm(dataset.sequence_list, desc=f"[{dataset_name} eval]")
            for seq_name in tbar:
                # 4. decide sampling strategy to choose sample frames, from all frames (seq_num_frames) of a sequence
                ids = seq_id_map[seq_name]

                # 5. load data sample (only extrinsics are used)
                batch = dataset.get_data(sequence_name=seq_name, ids=ids)
                gt_extrs = batch["extrs"]
                
                with torch.amp.autocast(device_type=hydra_cfg.device, dtype=torch.float64):
                    # 6. infer cameras
                    pred_extrs, pred_intrs = infer_cameras_w2c(batch['image_paths'], model, hydra_cfg)

                    # 7. compute metrics
                    rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(
                        pred_se3   = pred_extrs,
                        gt_se3     = gt_extrs,
                        # num_frames = num_frames,
                        num_frames = len(ids),
                    )

                # 8. update metric for a sequence
                tbar.set_postfix_str(f"Sequence {seq_name} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}")
                # model_logger.info(f"Sequence {seq_name} RotErr(Deg): {rel_rangle_deg.mean():5.2f} | TransErr(Deg): {rel_tangle_deg.mean():5.2f}")

                rError.extend(rel_rangle_deg.cpu().numpy())
                tError.extend(rel_tangle_deg.cpu().numpy())
            
            rError = np.array(rError)
            tError = np.array(tError)
            # 9. arrange all intermediate results to metrics
            for threshold in dataset_info.metric_thresholds:
                metric_dict[f"Racc_{threshold}"] = np.mean(rError < threshold).item() * 100
                metric_dict[f"Tacc_{threshold}"] = np.mean(tError < threshold).item() * 100
                Auc, _ = calculate_auc_np(rError, tError, max_threshold=threshold)
                metric_dict[f"Auc_{threshold}"]  = Auc.item() * 100

            model_logger.info(f"{dataset_name} - Average pose estimation metrics: {metric_dict}")

            # 9. save evaluation metrics to csv
            statistics_data = {"model": model_keyname, **metric_dict}
            statistics_file = osp.join(hydra_cfg.output_dir, f"{dataset_name}-metric")  # + ".csv"
            if getattr(hydra_cfg, "save_suffix", None) is not None:
                statistics_file += f"-{hydra_cfg.save_suffix}"
            statistics_file += ".csv"
            write_csv(statistics_file, statistics_data)

        del model
        torch.cuda.empty_cache()
        model_logger.info(f"Finished evaluating model {model_keyname} on all datasets.")


if __name__ == "__main__":
    set_default_arg("evaluation", "relpose-angular")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()
