# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2025 WindVChen.
# All rights reserved.
#
# This source code is derived from VGGT and licensed under the VGGT License
# found in the LICENSE_VGGT file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from dataclasses import dataclass
from posegam.utils.pose_enc import extri_intri_to_pose_encoding
from posegam.training.train_utils.general import check_and_fix_inf_nan


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module for PoseGAM. Combines the camera-pose loss and the mask loss.
    """
    def __init__(self, camera=None, mask=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.mask = mask

    def forward(self, predictions, batch) -> torch.Tensor:
        """
        Compute the total multi-task loss.

        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks

        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}

        # Camera pose loss - if pose encodings are predicted
        if "pose_enc_list" in predictions:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)

        # Mask reconstruction loss - if masks are predicted
        if "mask_recon" in predictions:
            mask_loss_dict = compute_mask_loss(predictions, batch, **self.mask)
            mask_loss = mask_loss_dict["loss_mask_dice"] + mask_loss_dict["loss_mask_bce"]
            mask_loss = mask_loss * self.mask["weight"]
            total_loss = total_loss + mask_loss
            loss_dict.update(mask_loss_dict)

        loss_dict["objective"] = total_loss

        return loss_dict


def compute_mask_loss(predictions, batch, **kwargs):
    """
    Compute mask loss.

    Args:
        predictions: Dict containing 'mask_recon'
    """
    pred_mask = predictions['mask_recon']

    gt_mask = batch['point_masks'].unsqueeze(2).float()  # (B, S, 1, H, W)
    gt_mask = check_and_fix_inf_nan(gt_mask, "gt_mask")

    if "loss_type" in kwargs:
        loss_type = kwargs["loss_type"]
    else:
        loss_type = "bce_dice"

    # camera_mask==0 gets full weight; camera_mask==1 gets reduced weight.
    camera_mask = batch['camera_mask']
    camera_mask_weight = float(kwargs.get("camera_mask_weight", 0.5))
    weight_mask = torch.where(
        camera_mask == 1,
        torch.full_like(camera_mask, camera_mask_weight, dtype=pred_mask.dtype),
        torch.ones_like(camera_mask, dtype=pred_mask.dtype),
    )
    while weight_mask.ndim < pred_mask.ndim:
        weight_mask = weight_mask.unsqueeze(-1)

    if loss_type == "l1":
        # For L1 loss, compute element-wise absolute difference
        loss = (pred_mask - gt_mask).abs()
        loss = check_and_fix_inf_nan(loss, "loss_Mask")
        loss = loss.clamp(max=100)
        loss = (loss * weight_mask).sum() / weight_mask.sum().clamp_min(1e-6)
    elif loss_type == "l2":
        # L2 norm for each component
        # pred.shape = B, S, C, H, W - compute norm along channel dimension
        loss = (pred_mask - gt_mask).norm(dim=2, keepdim=True)
        loss = check_and_fix_inf_nan(loss, "loss_Mask")
        loss = loss.clamp(max=100)
        loss = (loss * weight_mask).sum() / weight_mask.sum().clamp_min(1e-6)
    elif loss_type == "bce_dice":
        # BCEWithLogits expects raw logits; keep mask head output without sigmoid.
        bce = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction='none')

        # Soft Dice on probabilities for overlap optimization.
        pred_prob = torch.sigmoid(pred_mask)
        intersection = (pred_prob * gt_mask).sum(dim=(2, 3, 4))
        denominator = pred_prob.sum(dim=(2, 3, 4)) + gt_mask.sum(dim=(2, 3, 4))
        eps = float(kwargs.get("dice_eps", 1e-6))
        dice_loss = 1.0 - (2.0 * intersection + eps) / (denominator + eps)

        # Reduce BCE per frame, then combine with Dice and camera weights.
        bce_per_frame = bce.mean(dim=(2, 3, 4))
        loss_bce_weight = float(kwargs.get("loss_bce_weight", 1.0))
        loss_dice_weight = float(kwargs.get("loss_dice_weight", 1.0))
        per_frame_bce_loss = loss_bce_weight * bce_per_frame
        per_frame_dice_loss = loss_dice_weight * dice_loss

        frame_weight = weight_mask.squeeze(-1).squeeze(-1).squeeze(-1)
        bce_loss = (per_frame_bce_loss * frame_weight).sum() / frame_weight.sum().clamp_min(1e-6)
        dice_loss = (per_frame_dice_loss * frame_weight).sum() / frame_weight.sum().clamp_min(1e-6)
        bce_loss = check_and_fix_inf_nan(bce_loss, "loss_mask_bce")
        dice_loss = check_and_fix_inf_nan(dice_loss, "loss_mask_dice")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_dict = {
        f"loss_mask_bce": bce_loss,
        f"loss_mask_dice": dice_loss,
    }

    return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    gamma=0.6,              # temporal decay weight for multi-stage training
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    # List of predicted pose encodings per stage
    pred_pose_encodings = pred_dict['pose_enc_list']
    # Number of prediction stages
    n_stages = len(pred_pose_encodings)

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsics']
    gt_intrinsics = batch_data['intrinsics']
    image_hw = batch_data['images'].shape[-2:]

    camera_mask = batch_data['camera_mask']

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0

    # Compute loss for each prediction stage with temporal weighting
    for stage_idx in range(n_stages):
        # Later stages get higher weight (gamma^0 = 1.0 for final stage)
        stage_weight = gamma ** (n_stages - stage_idx - 1)
        pred_pose_stage = pred_pose_encodings[stage_idx]

        # Only consider valid frames for loss computation
        loss_T_stage, loss_R_stage, loss_FL_stage = camera_loss_single(
            pred_pose_stage.clone(),
            gt_pose_encoding.clone(),
            loss_type=loss_type,
            camera_mask=camera_mask
        )
        # Accumulate weighted losses across stages
        total_loss_T += loss_T_stage * stage_weight
        total_loss_R += loss_R_stage * stage_weight
        total_loss_FL += loss_FL_stage * stage_weight

    # Average over all stages
    avg_loss_T = total_loss_T / n_stages
    avg_loss_R = total_loss_R / n_stages
    avg_loss_FL = total_loss_FL / n_stages

    # Compute total weighted camera loss
    total_camera_loss = (
        avg_loss_T * weight_trans +
        avg_loss_R * weight_rot +
        avg_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": avg_loss_T,
        "loss_R": avg_loss_R,
        "loss_FL": avg_loss_FL
    }


def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1", camera_mask=None):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.

    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)

    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # if there is camera mask, downweight the known (camera_mask == 1) frames
    if camera_mask is not None:
        # Create weight mask: 0.5 for camera_mask == 1, 1.0 for others
        weight_mask = torch.where(camera_mask == 1, 0.5, 1.0)

        loss_T = loss_T * weight_mask.unsqueeze(-1)
        loss_R = loss_R * weight_mask.unsqueeze(-1)
        loss_FL = loss_FL * weight_mask.unsqueeze(-1)

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL
