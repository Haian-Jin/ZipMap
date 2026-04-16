# Code for ZipMap (CVPR 2026); created by Haian Jin

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any
import collections

from zipmap.layers import PatchEmbed
from zipmap.layers.block_ttt import Block
from zipmap.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from zipmap.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

TTTOperator = collections.namedtuple("TTTOperator", ["start", "end", "update", "apply"])




class Aggregator(nn.Module):
    """
    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        ttt_config=None,
        nvs_config=None,
        other_config=None,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None
        if ttt_config is not None:
            self.ttt_mode = ttt_config.get("ttt_mode", False)
            self.ttt_params = ttt_config.get("params", {})
        else:
            self.ttt_mode = False
            self.ttt_params = {}
        if other_config is not None:
            self.other_config = other_config
        else:
            self.other_config = {}
        self.affine_invariant = self.other_config.get("affine_invariant", False)

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    ttt_mode=False,
                    ttt_params={}
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                    ttt_mode=self.ttt_mode,
                    ttt_params=self.ttt_params
                )
                for _ in range(depth)
            ]
        )
        self.enable_NVS = nvs_config is not None
        if self.enable_NVS:
            ray_patch_size = patch_size
            ray_in_chans = nvs_config.get("nvs_ray_cond_dim", 6)

            nvs_input_type = nvs_config.get("nvs_input_type", "unposed_ray")
            self.nvs_input_type = nvs_input_type

            if nvs_input_type == "posed_ray":
                self.input_ray_patch_embed = PatchEmbed(
                        img_size=img_size,
                        patch_size=ray_patch_size,
                        in_chans=ray_in_chans,
                        embed_dim=embed_dim,
                        norm_layer=nn.LayerNorm,
                        )
                self.fused_layernorm = nn.LayerNorm(embed_dim)
            self.ray_patch_embed = PatchEmbed(
                                    img_size=img_size,
                                    patch_size=ray_patch_size,
                                    in_chans=ray_in_chans,
                                    embed_dim=embed_dim,
                                    norm_layer=nn.LayerNorm,
                                    # norm_layer=nn.Identity
                                    )
            self.ray_cond_token = nn.Parameter(torch.randn(1, 1, 1 + num_register_tokens, embed_dim))
            nn.init.normal_(self.ray_cond_token, std=1e-6)


        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size
        if self.affine_invariant:
            logging.info("Aggregator initialized with affine_invariant=True")
        else:
            logging.info("Aggregator initialized with affine_invariant=False")
        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, target_query_conditions: torch.Tensor = None, info: dict = {}) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        total_S = 0
        if images is not None:
            device = images.device
            B, S_input, C_in, H, W = images.shape

            if C_in != 3:
                raise ValueError(f"Expected 3 input channels, got {C_in}")

            # Normalize images and reshape for patch embed
            images = (images - self._resnet_mean) / self._resnet_std

            # Reshape to [B*S, C, H, W] for patch embedding
            images = images.view(B * S_input, C_in, H, W)
            patch_tokens = self.patch_embed(images)

            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]

            if self.enable_NVS and self.nvs_input_type == "posed_ray":
                input_ray_conditions = target_query_conditions["input_view_conditions"]
                input_ray_conditions = input_ray_conditions.view(B * S_input, input_ray_conditions.shape[2], H, W)
                input_ray_patch_tokens = self.input_ray_patch_embed(input_ray_conditions)
                patch_tokens = patch_tokens + input_ray_patch_tokens
                patch_tokens = self.fused_layernorm(patch_tokens)

            # Expand camera and register tokens to match batch size and sequence length
            camera_token = slice_expand_and_flatten(self.camera_token, B, S_input, affine_invariant=self.affine_invariant)
            register_token = slice_expand_and_flatten(self.register_token, B, S_input, affine_invariant=self.affine_invariant)

            # Concatenate special tokens with patch tokens
            tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
            total_S = total_S + S_input

        # If have query conditions for target views
        if target_query_conditions is not None and (target_query_conditions["query_conditions"] is not None):
            target_conditions = target_query_conditions["query_conditions"]
            device = target_conditions.device
            B, S_target, _, H, W = target_conditions.shape
            target_conditions = target_conditions.view(B * S_target, target_conditions.shape[2], H, W)
            target_patch_tokens = self.ray_patch_embed(target_conditions)
            ray_cond_special_token = self.ray_cond_token.squeeze(0).squeeze(0).expand(B * S_target, -1, -1)  # [B*S_target, 1 + num_register_tokens, C]
            target_ray_cond_tokens = torch.cat([ray_cond_special_token, target_patch_tokens], dim=1)  # [B*S_target, 1+num_register_tokens+P, C]
            C = target_ray_cond_tokens.shape[-1]
            # merge tokens and ray_cond_tokens to make them # [B*S_target, 1+num_register_tokens+P_per_image, C]
            target_ray_cond_tokens = target_ray_cond_tokens.view(B, S_target, -1, C)
            if images is not None:
                tokens = tokens.view(B, S_input, -1, C)
                tokens = torch.cat([tokens, target_ray_cond_tokens], dim=1)  # [B, S+S_target, ...]
            else:
                tokens = target_ray_cond_tokens  # [B, S_target, ...]
            total_S = total_S + S_target
            tokens = tokens.view(B * total_S, -1, C)

        # update P because we added special tokens
        _, P, C = tokens.shape
        
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * total_S, H // self.patch_size, W // self.patch_size, device=device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * total_S, self.patch_start_idx, 2).to(device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1) # [B*S, P, 2]



        if "ttt_op_order" not in info:
            # Scene State Query
            if target_query_conditions is not None:
                num_input_tokens = S_input * P
                ttt_op_order = [
                    TTTOperator(start=0, end=num_input_tokens, update=True, apply=False),
                    TTTOperator(start=0, end=-1, update=False, apply=True),
                ]
            # Don't do query.
            else:
                window_size = info.get("window_size", None)
                # Online reconstruction
                if window_size is not None and window_size > 0:
                    chunk_token_num = window_size * P
                    ttt_op_order = []
                    for start_idx in range(0, total_S * P, chunk_token_num):
                        end_idx = min(start_idx + chunk_token_num, total_S * P)
                        ttt_op_order.append(
                            TTTOperator(start=start_idx, end=end_idx, update=True, apply=False)
                        )
                        ttt_op_order.append(
                            TTTOperator(start=start_idx, end=end_idx, update=False, apply=True)
                        )
                # Bidirectional offline reconstruction
                else:
                    ttt_op_order = [
                        TTTOperator(start=0, end=-1, update=True, apply=False),
                        TTTOperator(start=0, end=-1, update=False, apply=True),
                    ]
            info["ttt_op_order"] = ttt_op_order

        frame_idx = 0
        global_idx = 0
        output_list = []
        used_by_DPT_idx  = [4, 11, 17, 23]
        state_list = []

        for cur_block_num in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, total_S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if "state_list" in info:
                        cur_cached_fast_weight = info["state_list"][cur_block_num]
                        info.update(cur_cached_fast_weight)
                    tokens, global_idx, global_intermediates, state = self._process_global_attention(
                        tokens, B, total_S, P, C, global_idx, pos=pos, info=info,
                    )
                    state_list.append(state)

                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # Save memory by only storing features needed by DPT head
            # This finding has also been adopted by prior work FastVGGT
            if cur_block_num in used_by_DPT_idx :
                for i in range(len(frame_intermediates)):
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
                    del concat_inter
        
        del frame_intermediates
        del global_intermediates

        return output_list, self.patch_start_idx, state_list



    def render(self, ray_conditions: torch.Tensor = None, info: dict = {}) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            ray_conditions (torch.Tensor): Input ray conditions with shape [B, S, nvs_ray_cond_dim, H, W].
                B: batch size, S: sequence length, nvs_ray_cond_dim: number of channels, H: height, W: width
        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = ray_conditions.shape

        # ray_conditions: [B, S_target, nvs_ray_cond_dim, H, W]
        S_target = ray_conditions.shape[1]
        ray_conditions = ray_conditions.view(B * S_target, ray_conditions.shape[2], H, W)
        ray_patch_tokens = self.ray_patch_embed(ray_conditions)
        C = ray_patch_tokens.shape[-1]
        ray_cond_special_token = self.ray_cond_token.squeeze(0).squeeze(0).expand(B * S_target, -1, -1)  # [B*S_target, 1 + num_register_tokens, C]
        ray_cond_tokens = torch.cat([ray_cond_special_token, ray_patch_tokens], dim=1)  # [B*S_target, 1+num_register_tokens+P, C]
        
        tokens = ray_cond_tokens

        total_S = S_target
        
        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * total_S, H // self.patch_size, W // self.patch_size, device=ray_conditions.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * total_S, self.patch_start_idx, 2).to(ray_conditions.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1) # [B*S, P, 2]
        
        # update P because we added special tokens
        _, P, C = tokens.shape


        frame_idx = 0
        global_idx = 0
        output_list = []
        used_by_DPT_idx  = [4, 11, 17, 23]
        state_list = []


        for cur_block_num in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, total_S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    cur_cached_fast_weight = info["state_list"][cur_block_num]
                    info.update(cur_cached_fast_weight)
                    tokens, global_idx, global_intermediates, state = self._process_global_attention(
                        tokens, B, total_S, P, C, global_idx, pos=pos, info=info,
                    )
                    state_list.append(state)

                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")


            if cur_block_num in used_by_DPT_idx :
                # save memory by only storing features needed by DPT head
                # This finding has also been adopted by prior work FastVGGT
                for i in range(len(frame_intermediates)):
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)
                    del concat_inter
        
        del frame_intermediates
        del global_intermediates

        return output_list, self.patch_start_idx, state_list


    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)
        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []
        info = {}
        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, _ = checkpoint(self.frame_blocks[frame_idx], tokens, pos, info,use_reentrant=self.use_reentrant)
            else:
                tokens, _ = self.frame_blocks[frame_idx](tokens, pos=pos, info=info)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None, info= {}):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []


        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens, state = checkpoint(self.global_blocks[global_idx], tokens, pos, info, use_reentrant=self.use_reentrant)
            else:
                tokens, state = self.global_blocks[global_idx](tokens, pos=pos, info=info)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates, state


def slice_expand_and_flatten(token_tensor, B, S, affine_invariant=False):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """
    if not affine_invariant:
        # Slice out the "query" tokens => shape (1, 1, ...)
        query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
        # Slice out the "other" tokens => shape (1, S-1, ...)
        others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
        # Concatenate => shape (B, S, ...)
        combined = torch.cat([query, others], dim=1)
    else:
        # For affine invariant, use the same token for all frames
        combined = token_tensor[:, 1:, ...].expand(B, S, *token_tensor.shape[2:])

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
