import hydra
import os
import os.path as osp
import torch
import logging
import json
from omegaconf import DictConfig, ListConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from utils.files import get_all_sequences, list_imgs_a_sequence
# from utils.debug import setup_debug
from utils.messages import set_default_arg
from videodepth.utils import save_depth_maps


@hydra.main(version_base="1.2", config_path="../configs", config_name="eval")
def main(hydra_cfg: DictConfig):
    # setup_debug(hydra_cfg.debug)
    logger = logging.getLogger("videodepth-infer")

    all_eval_models: ListConfig   = hydra_cfg.eval_models    # see configs/evaluation/videodepth.yaml
    all_eval_datasets: ListConfig = hydra_cfg.eval_datasets  # see configs/evaluation/videodepth.yaml
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

        # 0.3 look up infer_videodepth function
        infer_func_cfg = model_info.get(
            "infer_videodepth",
            DictConfig({
                '_target_': f'interfaces.{model_keyname}.infer_videodepth',
                '_partial_': True,
            })
        )
        infer_videodepth = hydra.utils.instantiate(infer_func_cfg)

        model_logger = logging.getLogger(f"videodepth-infer-{model_keyname}")
        for idx_dataset, dataset_name in enumerate(all_eval_datasets, start=1):
            # 1. look up dataset config from configs/data
            if dataset_name not in all_data_info:
                raise ValueError(f"Unknown dataset in global data information: {dataset_name}")
            dataset_info = all_data_info[dataset_name]
            max_n_frames = dataset_info.get("num_frames", None)

            # 2. get the sequence list
            if dataset_info.type == "video":
                # most of the datasets have many sequences of video
                seq_list = get_all_sequences(dataset_info)
            elif dataset_info.type == "mono":
                raise ValueError("dataset type `mono` is not supported for videodepth evaluation")
            else:
                raise ValueError(f"Unknown dataset type: {dataset_info.type}")

            model = model.eval()
            output_root = osp.join(hydra_cfg.output_dir, model_keyname, dataset_name)
            model_logger.info(f"[{idx_dataset}/{len(all_eval_datasets)}] Infering videodepth on {dataset_name} dataset..., output to {osp.relpath(output_root, hydra_cfg.work_dir)}")

            # 3. infer for each sequence (video)
            for seq_idx, seq in enumerate(seq_list, start=1):
                filelist = list_imgs_a_sequence(dataset_info, seq)
                if max_n_frames is not None:
                    filelist = filelist[:max_n_frames]
                save_dir = osp.join(output_root, seq)

                if not hydra_cfg.overwrite and (osp.isdir(save_dir) and len(os.listdir(save_dir)) == 2 * len(filelist) + 1):
                    model_logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} already processed, skipping.")
                    continue
                
                while True:
                    try:
                        # time_used: float, or List[float] (len = 2)
                        # depth_maps: (N, H, W), torch.Tensor
                        # conf_self: (N, H, W) torch.Tensor, or just None is ok
                        time_used, depth_maps, conf_self = infer_videodepth(filelist, model, hydra_cfg)
                        model_logger.info(f"[{seq_idx}/{len(seq_list)}] Sequence {seq} processed, time: {time_used}, saving depth maps...")

                        os.makedirs(save_dir, exist_ok=True)
                        save_depth_maps(depth_maps, save_dir, conf_self=conf_self)
                        # save time
                        with open(osp.join(save_dir, "_time.json"), "w") as f:
                            json.dump({
                                "time": time_used,
                                "frames": len(filelist),
                            }, f, indent=4)
                        break
                    except Exception as e:
                        
                        if "out of memory" in str(e):
                            # Handle OOM
                            torch.cuda.empty_cache()  # Clear the CUDA memory
                            model_logger.warning(f"OOM error in sequence {seq}, skipping this sequence.")
                        elif "Degenerate covariance rank" in str(e) or "Eigenvalues did not converge" in str(e):
                            # Handle Degenerate covariance rank exception and Eigenvalues did not converge exception
                            model_logger.warning(f"Exception in sequence {seq}: {str(e)}")
                            model_logger.warning(f"Traj evaluation error in sequence {seq}, skipping.")
                        else:
                            raise e  # Rethrow if it's not an expected exception
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    set_default_arg("evaluation", "videodepth")
    os.environ["HYDRA_FULL_ERROR"] = '1'
    with torch.no_grad():
        main()