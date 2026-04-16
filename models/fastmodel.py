import torch
import torch.nn as nn

from typing import Optional, Dict

import rootutils
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os

class Pi3(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.pi3.models.pi3 import Pi3 as Pi3Model
            self.model = Pi3Model.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor):
        return self.model(images)


class VGGT(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.vggt.models.vggt import VGGT as VGGTModel
            if pretrained_model_name_or_path.endswith('.pt') or pretrained_model_name_or_path.endswith('.pth'):
                self.model = VGGTModel()
                checkpoint = torch.load(pretrained_model_name_or_path, map_location="cpu", weights_only=True)
                model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)
                if missing:
                    print(f"Missing keys in state_dict: {missing}")
                if unexpected:
                    print(f"Unexpected keys in state_dict: {unexpected}")
                self.model.eval()
            else:
                self.model = VGGTModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images, query_points)
    

class MoGe(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            ori_model: bool = True,
        ):
        super().__init__()

        if ori_model and pretrained_model_name_or_path is not None:
            from models.moge.model.v1 import MoGeModel
            self.model = MoGeModel.from_pretrained(pretrained_model_name_or_path)
        else:
            raise NotImplementedError

    def forward(self, image: torch.Tensor, num_tokens: int) -> Dict[str, torch.Tensor]:
        return self.model(image, num_tokens)


class ZipMap(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
            ori_model: bool = True,
            affine_invariant: bool = True,
        ):
        super().__init__()


        model_config = {
            "img_size": 518,
            "patch_size": 14,
            "embed_dim": 1024,
            "enable_camera": True,
            "enable_local_point": False,
            "enable_depth": True,
            "ttt_config": 
                {
                    "ttt_mode": True,
                    "params": {
                        "bias": True,   
                        "head_dim": 1024,  # 256
                        "inter_multi": 2,
                        "base_lr": 0.01,
                        "muon_update_steps": 5,
                        "use_gate_fn": True
                    }
                },
            "other_config": {
                "use_gradient_checkpointing_local_point": False,
                "use_gradient_checkpointing_depth": False,
                "affine_invariant": affine_invariant, 
            }
        }

        if ori_model and pretrained_model_name_or_path is not None:
            from models.zipmap.models.ZipMap import ZipMap
            self.model = ZipMap(**model_config)
            if not os.path.exists(pretrained_model_name_or_path):
                print(f"Pretrained model path {pretrained_model_name_or_path} does not exist!")
            else:
                checkpoint = torch.load(pretrained_model_name_or_path, map_location="cpu",weights_only=True)
                model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
                missing, unexpected = self.model.load_state_dict(model_state_dict, strict=False)
                if missing:
                    print(f"Missing keys in state_dict: {missing}")
                if unexpected:
                    print(f"Unexpected keys in state_dict: {unexpected}")
                self.model.eval()

        else:
            raise NotImplementedError

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        return self.model(images, query_points)



class CUT3R(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.ttt3r.dust3r.model import ARCroco3DStereo
            self.model = ARCroco3DStereo.from_pretrained(pretrained_model_name_or_path)
            self.model.config.model_update_type = "cut3r"
        else:
            raise NotImplementedError

    @torch.no_grad()
    def inference_recurrent_lighter(self, groups, model, device, verbose=True):
        # if verbose:
        #     print(f">> Inference with model on {len(groups)} image/raymaps")

        with torch.cuda.amp.autocast(enabled=False):
            preds, batch, state_args = model.forward_recurrent_lighter(
                groups, device, ret_state=True
            )
            res = dict(views=batch, pred=preds)
        return res, state_args

    def forward(self, views):
        return self.inference_recurrent_lighter(
            views, self.model, device='cuda' if torch.cuda.is_available() else 'cpu'
        )


class TTT3R(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path: Optional[str] = None,
        ):
        super().__init__()

        if pretrained_model_name_or_path is not None:
            from models.ttt3r.dust3r.model import ARCroco3DStereo
            self.model = ARCroco3DStereo.from_pretrained(pretrained_model_name_or_path)
            self.model.config.model_update_type = "ttt3r"
        else:
            raise NotImplementedError

    @torch.no_grad()
    def inference_recurrent_lighter(self, groups, model, device, verbose=True):
        # if verbose:
        #     print(f">> Inference with model on {len(groups)} image/raymaps")

        with torch.cuda.amp.autocast(enabled=False):
            preds, batch, state_args = model.forward_recurrent_lighter(
                groups, device, ret_state=True
            )
            res = dict(views=batch, pred=preds)
        return res, state_args

    def forward(self, views):
        return self.inference_recurrent_lighter(
            views, self.model, device='cuda' if torch.cuda.is_available() else 'cpu'
        )



