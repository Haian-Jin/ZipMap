import torch
import cv2

from omegaconf import DictConfig

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from models.fastmodel import MoGe

def infer_monodepth(file: str, model: MoGe, hydra_cfg: DictConfig):
    device = hydra_cfg.device

    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    if hasattr(hydra_cfg, 'load_img_size'):
        resize_to = hydra_cfg.load_img_size
        height, width = min(resize_to, int(resize_to * height / width)), min(resize_to, int(resize_to * width / height))
        image = cv2.resize(image, (width, height), cv2.INTER_AREA)
    image_tensor = torch.tensor(image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

    # Inference
    output = model.model.infer(image_tensor, apply_mask=False)
    # points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
    return output['depth']