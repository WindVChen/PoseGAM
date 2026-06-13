# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2025 WindVChen.
# All rights reserved.
#
# This source code is derived from VGGT and licensed under the VGGT License
# found in the LICENSE_VGGT file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from posegam.layers import PatchEmbed
from posegam.layers.block import Block
from posegam.layers.attention import CrossAttention
from posegam.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from posegam.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

import math

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

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
        info_channels (int): Number of additional per-pixel info channels fed through the cross-attention context.
        stride (int): Downsampling stride used for the geometry (point-feature) patch embed.
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
        info_channels=0,
        stride=1,
    ):
        super().__init__()

        self.stride = stride
        self.embed_dim = embed_dim

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim, info_channels=info_channels, depth=depth)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

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
                )
                for _ in range(depth)
            ]
        )

        # Cross-attention blocks let the patch tokens attend to the per-pixel add-info context.
        self.cross_frame_blocks = nn.ModuleList(
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
                    attn_class=CrossAttention,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = num_register_tokens + 1

        # Initialize parameters with small values
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
        info_channels=0,
        depth=24,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """
        # Patch embed for the per-pixel Sonata geometry features (1088 channels).
        self.add_geo_patch_embed = PatchEmbed(img_size=img_size // self.stride, patch_size=14 // self.stride, in_chans=1088, embed_dim=embed_dim, norm_layer=nn.LayerNorm)

        if info_channels > 0:
            self.norm = nn.LayerNorm(embed_dim)
            self.add_info_patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=info_channels, embed_dim=embed_dim, norm_layer=nn.LayerNorm)
            self.unknown_add_info_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            self.add_info_pos_embed = nn.Parameter(torch.randn(1, self.add_info_patch_embed.num_patches, embed_dim))

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

    def forward(self, images: torch.Tensor, camera_pose_tokens: torch.Tensor, camera_mask, add_infos, only_save_last=False, point_feat_info=None) -> Tuple[List[torch.Tensor], int, int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            camera_pose_tokens (torch.Tensor): Camera tokens with shape [B, S, 1, C], where C is the embedding dimension.
            camera_mask (torch.Tensor): Camera mask with shape [B, S], where 1 indicates valid (known) cameras.
            add_infos (torch.Tensor): Per-pixel info channels with shape [B, S, info_channels, H, W], or None.
            point_feat_info (torch.Tensor): Per-pixel Sonata geometry features with shape [B, S, 1088, H/stride, W/stride], or None.

        Returns:
            (list[torch.Tensor], int, int):
                The list of outputs from the attention blocks, and the patch_start_idx
                indicating where patch tokens begin (returned twice for backward compatibility).
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        add_info_tokens = None
        if add_infos is not None:
            B_add, S_add, C_add, H_add, W_add = add_infos.shape
            # Reshape add_infos to [B*S, C_add, H_add, W_add] for patch embedding
            add_infos = add_infos.view(B_add * S_add, C_add, H_add, W_add)

            add_info_tokens = self.add_info_patch_embed(add_infos)
            if isinstance(add_info_tokens, dict):
                add_info_tokens = add_info_tokens["x_norm_patchtokens"]

            # Fuse the per-pixel Sonata geometry features into the add-info tokens.
            if point_feat_info is not None:
                add_point_feat = checkpoint(self.add_geo_patch_embed, point_feat_info.view(B_add * S_add, 1088, H_add // self.stride, W_add // self.stride), use_reentrant=self.use_reentrant)
                if isinstance(add_point_feat, dict):
                    add_point_feat = add_point_feat["x_norm_patchtokens"]
                add_info_tokens = add_info_tokens + add_point_feat

            # Ensure add_info_tokens has the same number of patches as the position embedding.
            npatch = add_info_tokens.shape[1]
            N = self.add_info_pos_embed.shape[1]
            if npatch == N and H_add == W_add:
                patch_pos_embed = self.add_info_pos_embed
            else:
                dim = add_info_tokens.shape[-1]
                h0 = H_add // self.patch_size
                w0 = W_add // self.patch_size
                M = int(math.sqrt(N))  # Recover the number of patches in each dimension
                assert N == M * M
                kwargs = {}
                kwargs["size"] = (h0, w0)
                patch_pos_embed = nn.functional.interpolate(
                    self.add_info_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
                    mode="bicubic",
                    antialias=True,
                    **kwargs,
                )
                assert (h0, w0) == patch_pos_embed.shape[-2:]
                patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

            # add position embedding for add_info_tokens
            add_info_tokens = add_info_tokens + patch_pos_embed

            # normalize the add_info_tokens
            add_info_tokens = self.norm(add_info_tokens)

            # mask the add_info_tokens with the unknown token for invalid cameras
            add_info_tokens = add_info_tokens * camera_mask.flatten().unsqueeze(-1).unsqueeze(-1) + self.unknown_add_info_token * (1 - camera_mask.flatten().unsqueeze(-1).unsqueeze(-1))

        _, P_patch, C = patch_tokens.shape

        # Expand register tokens to match batch size and sequence length
        register_token = self.register_token.expand(B, S, -1, -1)
        register_token = register_token.reshape(B * S, *register_token.shape[2:])

        # Concatenate special tokens with patch tokens
        camera_pose_tokens = camera_pose_tokens.view(B * S, 1, C)
        tokens = torch.cat([camera_pose_tokens, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for indx in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, add_info_tokens=add_info_tokens
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if not only_save_last and indx in [4, 11, 17, 23]:  # save intermediates for the DPT-style heads
                for i in range(len(frame_intermediates)):
                    # concat frame and global intermediates, [B x S x P x 2C]
                    concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                    output_list.append(concat_inter)

        if only_save_last:
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, add_info_tokens=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).reshape(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                if add_info_tokens is not None:
                    tokens = checkpoint(self.cross_frame_blocks[frame_idx], tokens, pos, add_info_tokens, pos[:, self.patch_start_idx:], use_reentrant=self.use_reentrant)
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                if add_info_tokens is not None:
                    tokens = self.cross_frame_blocks[frame_idx](tokens, pos=pos, context=add_info_tokens, context_pos=pos[:, self.patch_start_idx:])
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens[:, :P].view(B, S, P, C))

        return tokens[:, :P], frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).reshape(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens[:, :S * P, :].view(B, S, P, C))
        # After processing, we only keep the first S*P tokens for the next iteration

        return tokens[:, :S * P, :].contiguous(), global_idx, intermediates
