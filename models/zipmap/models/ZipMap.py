import collections
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from zipmap.models.aggregator_ttt import Aggregator
from zipmap.heads.camera_head import CameraHead
from zipmap.heads.dpt_head_vggt_legacy import DPTHead

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])


class ZipMap(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True,
                 enable_camera_svd=False,
                 enable_local_point=True,
                 enable_depth=True,
                 enable_nvs=False,
                 ttt_config=None,
                 nvs_config=None,
                 other_config=None,
                 ):
        super().__init__()
        self.ttt_config = ttt_config
        self.other_config = other_config
        self.nvs_config = nvs_config
        if self.other_config is None:
            self.other_config = {}

        affine_invariant = self.other_config.get("affine_invariant", False)
        mixed_image_tokenization = self.other_config.get("mixed_image_tokenization", False)

        other_config_for_aggregator = {
            "affine_invariant": affine_invariant,
            "mixed_image_tokenization": mixed_image_tokenization,
        }

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, ttt_config=ttt_config, other_config=other_config_for_aggregator, nvs_config=nvs_config)

        if enable_camera:
            self.camera_head = CameraHead(dim_in=2 * embed_dim)
        else:
            self.camera_head = None

        if enable_camera_svd:
            # output is c2w 3 for translation, 9 for SVD rotation, 2 for FoV
            self.camera_svd_head = CameraHead(dim_in=2 * embed_dim, pose_encoding_type='absT_svdR_FoV')
        else:
            self.camera_svd_head = None

        if enable_local_point:
            use_gradient_checkpointing_local_point = self.other_config.get("use_gradient_checkpointing_local_point", False)
            self.local_point_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=3,
                activation="xy_exp",
                conf_activation="none", # no confidence for local points
                use_gradient_checkpointing=use_gradient_checkpointing_local_point,
                 # ! vggt set this to True but you should manually check set it to be False otherwise dpt residual block will have bugs
                use_inplace=False
            )
        else:
            self.local_point_head = None
        if enable_depth:
            use_gradient_checkpointing_depth = self.other_config.get("use_gradient_checkpointing_depth", False)
            self.depth_head = DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
                use_gradient_checkpointing=use_gradient_checkpointing_depth,
                # ! vggt set this to True but you should manually check set it to be False otherwise dpt residual block will have bugs
                use_inplace=False
            )
        else:
            self.depth_head = None

        if enable_nvs:
            use_gradient_checkpointing_nvs = self.other_config.get("use_gradient_checkpointing_nvs", False)
            nvs_input_type = self.nvs_config.get("nvs_input_type", "unposed_ray")
            nvs_output_type = self.nvs_config.get("nvs_output_type", "rgb")
            self.nvs_ray_cond_type = self.nvs_config.get("nvs_ray_cond_type", "default_plucker")
            self.nvs_input_type = nvs_input_type
            self.nvs_output_type = nvs_output_type
            if nvs_output_type == "rgb":
                self.nvs_head = DPTHead(
                    dim_in=2 * embed_dim,
                    output_dim=3,
                    activation="sigmoid",
                    conf_activation="none", # no confidence for rgb
                    use_gradient_checkpointing=use_gradient_checkpointing_nvs,
                    use_inplace=False
                )
            elif nvs_output_type == "rgbd":
                self.nvs_head = DPTHead(
                    dim_in=2 * embed_dim,
                    output_dim=5,
                    activation="sigmoid_exp",
                    conf_activation="expp1",
                    use_gradient_checkpointing=use_gradient_checkpointing_nvs,
                    use_inplace=False
                )
        else:
            self.nvs_head = None


    def forward(self, images: torch.Tensor, query_info: torch.Tensor = None, store_state: bool = False):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_info (torch.Tensor, optional): Query information for NVS,
                Default: None

        """        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)
        
        if query_info is not None and self.nvs_head is not None:
            aggregator_query_conditions = self.get_aggregator_query_conditions(query_info)
        else:
            aggregator_query_conditions = None

        info = {
            # "ttt_op_order": ttt_op_order, # define later in forward
            "store_state": store_state, # if to store the state_list for future queries
        }

        aggregated_tokens_list, patch_start_idx, state_list = self.aggregator(images, target_query_conditions=aggregator_query_conditions, info=info)
        
        input_view_num = images.shape[1]
        input_img_aggregated_tokens_list = [tokens[:, :input_view_num, :] for tokens in aggregated_tokens_list]

        predictions = {}
        with torch.amp.autocast(device_type='cuda', enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(input_img_aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list # var type: a list of predictions["pose_enc"] of each iteration
            

            if self.camera_svd_head is not None:
                pose_enc_svd_list = self.camera_svd_head(input_img_aggregated_tokens_list)
                predictions["pose_enc_svd"] = pose_enc_svd_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_svd_list"] = pose_enc_svd_list # var type: a list of predictions["pose_enc"] of each iteration
            
            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    input_img_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
    
            if self.local_point_head is not None:
                pts3d, pts3d_conf = self.local_point_head(
                    input_img_aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["local_points"] = pts3d
                predictions["local_points_conf"] = pts3d_conf


            predictions["images"] = images  # store the images for visualization during inference

        predictions["state_list"] = state_list # cache the TTT state_list for future queries
        
        
        # query the model with the conditions computed from predictions
        if query_info is not None and self.nvs_head is not None:

            nvs_aggregated_tokens_list = [tokens[:, input_view_num:, :] for tokens in aggregated_tokens_list]
            with torch.amp.autocast(device_type='cuda', enabled=False):
                # Create dummy images tensor with correct shape for target views
                B, _, C, H, W = images.shape
                S_target = query_info["target_view_num"]
                nvs_dummy_images = torch.zeros(B, S_target, C, H, W, dtype=images.dtype, device=images.device)
                
                nvs, nvs_conf = self.nvs_head(
                    nvs_aggregated_tokens_list, images=nvs_dummy_images, patch_start_idx=patch_start_idx
                )
                predictions["nvs_pred"] = nvs
                predictions["nvs_depth_conf"] = nvs_conf


        return predictions

    def get_aggregator_query_conditions(self, query_info):
        input_view_conditions = None
        query_conditions = None
        if self.nvs_input_type == "unposed_ray":
            query_conditions = query_info["nvs_target_ray_cond"]
        elif self.nvs_input_type == "posed_ray":
            input_view_conditions = query_info.get("input_ray_cond", None)
            query_conditions = query_info["nvs_target_ray_cond"]

        aggregator_query_conditions = {
            "cond_type": self.nvs_input_type,
            "query_conditions": query_conditions,
            "input_view_conditions": input_view_conditions,
        }
        return aggregator_query_conditions
    

    

    def render(self, info={}, ray_conditions: torch.Tensor = None, chunksize: int = 50):
        """
        render nvs rgb and depth given ray conditions only
        """
        B, S_target, C, H, W = ray_conditions.shape
        NVS_prediction_list = []
        NVS_prediction_conf_list = []
        for cur_idx in range(0, S_target, chunksize):
            end_idx = min(cur_idx + chunksize, S_target)
            cur_ray_conditions = ray_conditions[:, cur_idx:end_idx, :, :, :]
            aggregated_tokens_list, patch_start_idx, state_list = self.aggregator.render(ray_conditions=cur_ray_conditions, info=info)

            predictions = {}
            with torch.amp.autocast(device_type='cuda', enabled=False):
                # Create dummy images tensor with correct shape for target views
                nvs_dummy_images = torch.zeros(B, end_idx - cur_idx, C, H, W, dtype=ray_conditions.dtype, device=ray_conditions.device)
                nvs, nvs_conf = self.nvs_head(
                    aggregated_tokens_list, images=nvs_dummy_images, patch_start_idx=patch_start_idx
                )
                NVS_prediction_list.append(nvs)
                NVS_prediction_conf_list.append(nvs_conf)
        predictions["nvs_pred"] = torch.cat(NVS_prediction_list, dim=1)
        predictions["nvs_depth_conf"] = torch.cat(NVS_prediction_conf_list, dim=1)

        return predictions