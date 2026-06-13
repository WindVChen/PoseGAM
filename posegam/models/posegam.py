# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2025 WindVChen.
# All rights reserved.
#
# This source code is derived from VGGT and licensed under the VGGT License
# found in the LICENSE_VGGT file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from posegam.models.aggregator import Aggregator
from posegam.heads.camera_head import CameraHead, CameraEncode
from posegam.heads.dpt_head import DPTHead

from posegam.dependency import sonata


def downsample_mask(mask, stride, method="stride"):
    """
    Downsample mask using either stride or torch interpolation method.

    Args:
        mask (torch.Tensor): Input mask with shape [H, W]
        stride (int): Downsampling stride
        method (str): Either "stride" or "torch_interp"

    Returns:
        torch.Tensor: Downsampled mask
    """
    if method == "stride":
        return mask[..., ::stride, ::stride]
    elif method == "torch_interp":
        H, W = mask.shape[-2:]
        target_H = H // stride
        target_W = W // stride

        # Convert to float and add batch and channel dimensions for interpolation
        mask_tensor = mask.float().unsqueeze(0)  # [1, 1, H, W]

        # Perform bilinear interpolation
        downsampled_mask_tensor = F.interpolate(
            mask_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False
        )

        # Remove batch and channel dimensions and thresholdize to ensure boolean type
        downsampled_mask = downsampled_mask_tensor.squeeze(0) > 0.5

        return downsampled_mask
    else:
        raise ValueError(f"Unknown downsampling method: {method}")


class PoseGAM(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_mask=True, input_keys=None, stride=7, downsample_method="stride"):
        super().__init__()

        self.enable_camera = enable_camera
        self.enable_mask = enable_mask

        self.unknown_camera_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        nn.init.normal_(self.unknown_camera_token, std=1e-6)

        self.camera_encode = CameraEncode(target_dim=embed_dim)

        self.stride = stride
        self.downsample_method = downsample_method
        if self.downsample_method not in ["stride", "torch_interp"]:
            raise ValueError(f"downsample_method must be either 'stride' or 'torch_interp', got: {self.downsample_method}")

        # Additional per-pixel info channels concatenated to the image input.
        info_channel_dict = {"world_points": 3, "point_masks": 1, "base_colors": 3}
        if input_keys is not None and len(input_keys) > 0:
            for key in input_keys:
                if key not in info_channel_dict:
                    raise ValueError(f"Invalid input_key: {key}. Valid keys are: {list(info_channel_dict.keys())}")
            info_channels = sum(info_channel_dict[key] for key in input_keys)
        else:
            info_channels = 0
        self.input_keys = input_keys

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim, info_channels=info_channels, stride=stride)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.mask_head = DPTHead(dim_in=2 * embed_dim, output_dim=1, activation="linear", conf_activation="linear") if enable_mask else None

        # Sonata point-cloud encoder producing per-point geometry features.
        sonata.utils.set_seed(53124)
        self.shape_model = sonata.load("sonata", repo_id="facebook/sonata", pretrained=True)
        if hasattr(self.shape_model.embedding, "mask_token"):
            self.shape_model.embedding.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor, camera_params: torch.Tensor, camera_mask: torch.Tensor,
                world_points, point_masks, base_colors, point_sonata, initial_sonata_num) -> dict:
        """
        Forward pass of the PoseGAM model.

        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            camera_params (torch.Tensor): Camera parameters with shape [B, S, 9].
            camera_mask (torch.Tensor): Camera mask with shape [B, S], where 1 indicates valid (known) cameras.
            world_points (torch.Tensor): Per-pixel world coordinates with shape [B, S, 3, H, W].
            point_masks (torch.Tensor): Per-pixel validity masks with shape [B, S, 1, H, W].
            base_colors (torch.Tensor): Per-pixel base colors with shape [B, S, 3, H, W].
            point_sonata: Batched point-cloud input for the Sonata encoder.
            initial_sonata_num (torch.Tensor): Per-sample number of input points before Sonata transforms.
        Returns:
            dict: A dictionary containing the model predictions:
                - pose_enc (torch.Tensor): Camera pose encoding from the last iteration.
                - pose_enc_list (list): Camera pose encodings from all iterations.
                - mask_recon (torch.Tensor): Predicted masks.
                - images (torch.Tensor): The composited input images, preserved for visualization.
        """
        # For known cameras (camera_mask==1) feed the clean base colors, otherwise the input images.
        images = base_colors * camera_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + images * (1 - camera_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))

        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        # Concatenate the selected per-pixel info channels according to input_keys.
        if self.input_keys is not None and len(self.input_keys) > 0:
            add_infos = []
            for key in self.input_keys:
                if key == "world_points":
                    add_infos.append(world_points)
                elif key == "point_masks":
                    add_infos.append(point_masks)
                elif key == "base_colors":
                    add_infos.append(base_colors)
            add_infos = torch.cat(add_infos, dim=2)
        else:
            add_infos = None

        camera_tokens = self.camera_encode(camera_params)

        # For invalid cameras, replace the camera token with the learnable unknown-camera token.
        camera_tokens = camera_tokens * camera_mask.unsqueeze(-1) + self.unknown_camera_token * (1 - camera_mask.unsqueeze(-1))

        # Encode the point cloud with Sonata and propagate features back to the input resolution.
        point = self.shape_model(point_sonata)
        # Upcast point features: concatenate features across the two coarsest pooling levels.
        # `point` is a structure carrying all per-level information from the forward pass.
        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent

        # Propagate features down to the finest level.
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        batch_indices = point.batch
        batch_changes = torch.cat([torch.tensor([0], device=batch_indices.device),
                                (batch_indices[1:] != batch_indices[:-1]).nonzero(as_tuple=True)[0] + 1,
                                torch.tensor([len(batch_indices)], device=batch_indices.device)])
        batch_counts = batch_changes[1:] - batch_changes[:-1]

        point_feats = [point.feat[batch_changes[i]:batch_changes[i+1]] for i in range(len(batch_counts))]
        accum_num = torch.cumsum(initial_sonata_num, dim=0)
        accum_num = torch.cat([torch.tensor([0], device=accum_num.device), accum_num])
        point_inverses = [point.inverse[accum_num[i]:accum_num[i+1]] for i in range(len(accum_num)-1)]
        point_inverse_feats = [feats[inverse] for feats, inverse in zip(point_feats, point_inverses)]

        # Scatter the per-point features back onto the (downsampled) pixel grid.
        B, S, _, H, W = world_points.shape
        feat_dim = point_inverse_feats[0].shape[-1]
        zero_feat = torch.zeros(B, S, H // self.stride, W // self.stride, feat_dim, device=world_points.device)

        for batch_idx in range(B):
            batch_point_feats = point_inverse_feats[batch_idx]  # [N_points, feat_dim]
            # Sonata points are extracted from the first 10 frames of each sequence.
            for seq_idx in range(10):
                mask = downsample_mask(point_masks[batch_idx, seq_idx], self.stride, self.downsample_method)  # [H, W]
                valid_mask = mask.squeeze(0) > 0
                valid_indices = valid_mask.nonzero(as_tuple=False)  # [N_valid, 2]
                zero_feat[batch_idx, seq_idx, valid_indices[:, 0], valid_indices[:, 1]] = batch_point_feats[:valid_indices.shape[0]]
                batch_point_feats = batch_point_feats[valid_indices.shape[0]:]

        # [B, S, feat_dim, H, W], compatible with the add_infos format.
        point_feat_info = zero_feat.permute(0, 1, 4, 2, 3)

        aggregated_tokens_list, patch_start_idx, addinfo_patch_start_idx = self.aggregator(
            images, camera_tokens, camera_mask, add_infos,
            only_save_last=((self.camera_head is None) and (self.mask_head is None)),
            point_feat_info=point_feat_info,
        )

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list

            if self.mask_head is not None:
                tokens_list = [tokens[:, :, addinfo_patch_start_idx:] for tokens in aggregated_tokens_list]
                mask = self.mask_head(
                    tokens_list, images=images, patch_start_idx=0
                )
                predictions["mask_recon"] = mask.permute(0, 1, 4, 2, 3)

        predictions["images"] = images

        return predictions
