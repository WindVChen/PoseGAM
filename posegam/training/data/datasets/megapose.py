# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modifications Copyright (c) 2025 WindVChen.
# All rights reserved.
#
# This source code is derived from VGGT and licensed under the VGGT License
# found in the LICENSE_VGGT file in the root directory of this source tree.

import json
import os.path as osp
import os
import logging
from PIL import Image

import cv2
import random
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from posegam.dependency import sonata
import torch.nn.functional as F

from posegam.training.data.dataset_util import *
from posegam.training.data.base_dataset import BaseDataset
from .mask_gen import random_mask, free_form_mask, prepare_mask_and_masked_image

import imgaug as ia
import imgaug.augmenters as iaa
from pycocotools import mask as maskUtils

from posegam.evaluation.fix_BOP_translation import load_mesh_with_textures
import trimesh

def _analytical_pose_init(K_src, K_dst, R_src, T_src):
    """
    Compute an analytical initial estimate for (R_dst, T_dst) using
    the projection matrix approach with SVD polar decomposition.
    This is used as initialization for the PnP solver.
    """
    P_src = K_src @ np.hstack((R_src, T_src.reshape(3, 1)))  # 3x4
    M = np.linalg.inv(K_dst) @ P_src  # 3x4
    M33 = M[:, :3]
    m4 = M[:, 3]

    try:
        U, S, Vt = np.linalg.svd(M33)
    except np.linalg.LinAlgError:
        return R_src.copy(), T_src.copy()

    R_init = U @ Vt
    if np.linalg.det(R_init) < 0:
        U[:, -1] *= -1
        R_init = U @ Vt

    scale = np.mean(S[:2])
    T_init = m4 / (scale + 1e-8)
    return R_init, T_init

def convert_pose_between_intrinsics(
    K_src,
    K_dst,
    R_src,
    T_src,
    model_points=None,
):
    """
    Convert a pose (R_src, T_src) estimated under intrinsics K_src
    to the corresponding pose (R_dst, T_dst) under another intrinsic matrix K_dst,
    ensuring that 3D points project to the same 2D pixel locations.

    Uses PnP (Perspective-n-Point) solver with analytical initialization.
    The 3D test points MUST match the spatial scale of actual scene content
    (e.g., mesh vertices) — otherwise PnP fits a tiny angular patch and
    the solution diverges at the real object locations.

    Parameters
    ----------
    K_src : np.ndarray (3x3)
        Intrinsic matrix used during pose estimation.
    K_dst : np.ndarray (3x3)
        Target intrinsic matrix to convert the pose to.
    R_src : np.ndarray (3x3)
        Rotation matrix under K_src.
    T_src : np.ndarray (3,)
        Translation vector under K_src.
    model_points : np.ndarray (N, 3), optional
        Actual mesh vertices (or representative subset) in model/object space.
        If provided, PnP is solved at the exact locations that the renderer
        will use, giving the best possible mask alignment.
        If None, synthetic points are generated at a scale derived from T_src.

    Returns
    -------
    R_dst : np.ndarray (3x3)
        Converted rotation under K_dst.
    T_dst : np.ndarray (3,)
        Converted translation under K_dst.
    """

    # ---- Step 1: Gather 3D test points at the correct scale ----
    if model_points is not None:
        # Use actual mesh vertices — subsample if too many for speed
        if hasattr(model_points, 'cpu'):
            model_points = model_points.cpu().numpy()
        model_points = model_points.astype(np.float64)
        # if len(model_points) > 2000:
        #     idx = np.random.default_rng(42).choice(len(model_points), 2000, replace=False)
        #     pts_3d = model_points[idx]
        # else:
        pts_3d = model_points.copy()
    else:
        # Generate synthetic points scaled to match T_src magnitude
        # Object extent ~ ||T_src|| * tan(FOV/2); conservative: use ||T_src|| as scale
        scale = 100 #max(np.linalg.norm(T_src), 1.0)

        # 3D grid spanning [-scale, scale]
        lin = np.linspace(-scale, scale, 10)
        xx, yy, zz = np.meshgrid(lin, lin, lin)
        grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

        # Fibonacci sphere at radius = scale
        num_sphere = 200
        indices = np.arange(num_sphere, dtype=float) + 0.5
        phi_s = np.arccos(1 - 2 * indices / num_sphere)
        theta_s = np.pi * (1 + 5**0.5) * indices
        sphere_pts = scale * np.stack([
            np.sin(phi_s) * np.cos(theta_s),
            np.sin(phi_s) * np.sin(theta_s),
            np.cos(phi_s)
        ], axis=1)

        pts_3d = np.vstack([grid_pts, sphere_pts]).astype(np.float64)

    # ---- Step 1.5: Convert pose into transformed frame ----
    coord_transform = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],  # Y <- Z
        [0, -1,  0,  0],  # Z <- -Y
        [0,  0,  0,  1]
    ], dtype=np.float64)

    src_matrix = np.eye(4, dtype=np.float64)
    src_matrix[:3, :3] = R_src.astype(np.float64)
    src_matrix[:3, 3] = T_src.astype(np.float64)
    src_matrix_transformed = src_matrix #@ np.linalg.inv(coord_transform)
    R_src_transformed = src_matrix_transformed[:3, :3]
    T_src_transformed = src_matrix_transformed[:3, 3]

    # ---- Step 2: Transform to camera space and filter positive depth ----
    pts_cam = (R_src_transformed @ pts_3d.T + T_src_transformed.reshape(3, 1)).T  # [N, 3]
    mask = pts_cam[:, 2] > 1e-3
    pts_3d = pts_3d[mask]
    pts_cam = pts_cam[mask]

    if len(pts_3d) < 6:
        print("Warning: Too few points with positive depth, returning original pose")
        return R_src, T_src

    # ---- Step 3: Project to target 2D under transformed (K_src, R_src, T_src) ----
    pts_proj = (K_src @ pts_cam.T).T  # [N, 3]
    pts_2d = (pts_proj[:, :2] / pts_proj[:, 2:3]).astype(np.float64)  # [N, 2]

    # ---- Step 4: Get analytical initialization ----
    R_init, T_init = _analytical_pose_init(K_src, K_dst, R_src_transformed, T_src_transformed)
    rvec_init, _ = cv2.Rodrigues(R_init.astype(np.float64))
    tvec_init = T_init.reshape(3, 1).astype(np.float64)

    # ---- Step 5: Solve PnP with analytical initialization ----
    success, rvec, tvec = cv2.solvePnP(
        pts_3d.reshape(-1, 1, 3),
        pts_2d.reshape(-1, 1, 2),
        K_dst.astype(np.float64),
        None,
        rvec=rvec_init.copy(),
        tvec=tvec_init.copy(),
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        # Fallback: try SQPNP without initial guess
        success, rvec, tvec = cv2.solvePnP(
            pts_3d.reshape(-1, 1, 3),
            pts_2d.reshape(-1, 1, 2),
            K_dst.astype(np.float64),
            None,
            flags=cv2.SOLVEPNP_SQPNP
        )

    if not success:
        raise ValueError("Warning: PnP failed completely, using analytical initialization")

    # ---- Step 6: Refine with Levenberg-Marquardt ----
    try:
        rvec, tvec = cv2.solvePnPRefineLM(
            pts_3d.reshape(-1, 1, 3),
            pts_2d.reshape(-1, 1, 2),
            K_dst.astype(np.float64),
            None,
            rvec.copy(),
            tvec.copy(),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-8)
        )
    except Exception:
        pass  # LM refinement is optional; PnP result is already good

    R_dst_transformed, _ = cv2.Rodrigues(rvec)
    T_dst_transformed = tvec.flatten()

    # ---- Step 6.5: Transform optimized pose back to original frame before returning ----
    dst_matrix_transformed = np.eye(4, dtype=np.float64)
    dst_matrix_transformed[:3, :3] = R_dst_transformed.astype(np.float64)
    dst_matrix_transformed[:3, 3] = T_dst_transformed.astype(np.float64)
    dst_matrix = dst_matrix_transformed #@ coord_transform
    R_dst = dst_matrix[:3, :3]
    T_dst = dst_matrix[:3, 3]

    # ---- Step 8: Compute and log reprojection error ----
    pts_reproj = (K_dst @ (R_dst @ pts_3d.T + T_dst.reshape(3, 1))).T
    pts_reproj_2d = pts_reproj[:, :2] / pts_reproj[:, 2:3]
    reproj_err = np.mean(np.linalg.norm(pts_2d - pts_reproj_2d, axis=1))
    max_err = np.max(np.linalg.norm(pts_2d - pts_reproj_2d, axis=1))
    print(f"  Intrinsic conversion: mean reproj err = {reproj_err:.4f} px, max = {max_err:.4f} px  ({len(pts_3d)} pts)")

    return R_dst, T_dst

def parse_camera_parameters(frame_data, image_width=512, image_height=512):
    """
    Returns:
        intrinsics: 3x3 camera intrinsic matrix
        extrinsics: 4x4 extrinsic matrix for the view
    """

    # Parse intrinsics from first frame (same for all frames)
    first_frame = frame_data
    fov_x = first_frame['camera_angle_x']  # Field of view in radians
    
    # Calculate focal length from FOV
    # focal_length = image_width / (2 * tan(fov_x / 2))
    focal_length = image_width / (2.0 * np.tan(fov_x / 2.0))
    
    # Construct intrinsic matrix (assuming principal point at image center)
    intrinsics = np.array([
        [focal_length, 0, image_width / 2.0],
        [0, focal_length, image_height / 2.0],
        [0, 0, 1]
    ])
    
    # Transform matrix is already in the correct 4x4 format
    transform_matrix = np.array(frame_data['transform_matrix'])
        
    return intrinsics, np.array(transform_matrix)

def intrinsics_to_projection(
        intrinsics: torch.Tensor,
        near: float = 0.1,
        far: float = 1000.0,
    ) -> torch.Tensor:
    """
    OpenCV intrinsics to OpenGL perspective matrix

    Args:
        intrinsics (torch.Tensor): [3, 3] OpenCV intrinsics matrix
        near (float): near plane to clip
        far (float): far plane to clip
    Returns:
        (torch.Tensor): [4, 4] OpenGL perspective matrix
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = 2 * fx
    ret[1, 1] = 2 * fy
    ret[0, 2] = 2 * cx - 1
    ret[1, 2] = - 2 * cy + 1
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def farthest_point_sampling(points, num_samples):
    """
    Farthest Point Sampling (FPS) algorithm to select diverse points.
    
    Args:
        points (np.ndarray): Array of shape [N, 3] containing 3D points (camera positions)
        num_samples (int): Number of points to sample
    
    Returns:
        np.ndarray: Indices of selected points
    """
    if num_samples >= len(points):
        return np.arange(len(points))
    
    if num_samples <= 0:
        return np.array([], dtype=int)
    
    # Initialize with a random starting point
    selected_indices = [np.random.randint(0, len(points))]
    distances = np.full(len(points), np.inf)
    
    for _ in range(1, num_samples):
        # Update distances to the nearest selected point
        last_selected = points[selected_indices[-1]]
        current_distances = np.linalg.norm(points - last_selected, axis=1)
        distances = np.minimum(distances, current_distances)
        
        # Select the point that is farthest from all selected points
        next_idx = np.argmax(distances)
        selected_indices.append(next_idx)
    
    return np.array(selected_indices)


def generate_random_rotation(max_angle_degrees=30):
    """
    Generate a random 3D rotation matrix.
    
    Args:
        max_angle_degrees (float): Maximum rotation angle in degrees for each axis.
    
    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix with random rotation.
    """
    # Generate random rotation angles for each axis
    angles = np.random.uniform(-max_angle_degrees, max_angle_degrees, 3)
    # angles = np.array([90.0, 90.0, 90.0])  # For testing, use fixed angles

    # Create rotation object and get rotation matrix
    rotation = R.from_euler('xyz', angles, degrees=True)
    rotation_matrix = rotation.as_matrix()
    
    # Create 4x4 homogeneous transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    
    return transform_matrix


def downsample_point_arrays(world_points, normals, base_colors, point_masks, stride):
    """
    Downsample world points, normals, base colors, and point masks to target resolution.
    
    Args:
        world_points (np.ndarray): World coordinates array with shape (H, W, 3)
        normals (np.ndarray): Normal vectors array with shape (H, W, 3)
        base_colors (np.ndarray): Base color array with shape (H, W, 3)
        point_masks (np.ndarray): Point mask array with shape (H, W)
    
    Returns:
        tuple: Downsampled (world_points, normals, base_colors, point_masks)
    """    
    # Calculate stride for downsampling
    stride_h = stride
    stride_w = stride
    
    # Downsample by taking every stride-th pixel
    downsampled_world_points = world_points[::stride_h, ::stride_w]
    downsampled_normals = normals[::stride_h, ::stride_w]
    downsampled_base_colors = base_colors[::stride_h, ::stride_w]
    downsampled_point_masks = point_masks[::stride_h, ::stride_w]
    
    return downsampled_world_points, downsampled_normals, downsampled_base_colors, downsampled_point_masks


def downsample_point_arrays_torch_interp(world_points, normals, base_colors, point_masks, stride):
    """
    Downsample world points, normals, base colors, and point masks using PyTorch interpolation.
    
    Args:
        world_points (np.ndarray): World coordinates array with shape (H, W, 3)
        normals (np.ndarray): Normal vectors array with shape (H, W, 3)
        base_colors (np.ndarray): Base color array with shape (H, W, 3)
        point_masks (np.ndarray): Point mask array with shape (H, W)
        stride (int): Downsampling stride
    
    Returns:
        tuple: Downsampled (world_points, normals, base_colors, point_masks)
    """    
    H, W = world_points.shape[:2]
    target_H = H // stride
    target_W = W // stride
    
    # Convert numpy arrays to torch tensors and rearrange dimensions for interpolation
    # PyTorch interpolation expects (N, C, H, W) format
    world_points_tensor = torch.from_numpy(world_points).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    normals_tensor = torch.from_numpy(normals).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    base_colors_tensor = torch.from_numpy(base_colors).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    point_masks_tensor = torch.from_numpy(point_masks.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Perform bilinear interpolation with align_corners=False
    downsampled_world_points_tensor = F.interpolate(
        world_points_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False
    )
    downsampled_normals_tensor = F.interpolate(
        normals_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False
    )
    downsampled_base_colors_tensor = F.interpolate(
        base_colors_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False
    )
    downsampled_point_masks_tensor = F.interpolate(
        point_masks_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False
    )
    
    # Convert back to numpy arrays and rearrange dimensions
    downsampled_world_points = downsampled_world_points_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H', W', 3)
    downsampled_normals = downsampled_normals_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H', W', 3)
    downsampled_base_colors = downsampled_base_colors_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H', W', 3)
    downsampled_point_masks_raw = downsampled_point_masks_tensor.squeeze(0).squeeze(0).numpy()  # (H', W')
    
    # Thresholdize the mask to ensure boolean type (threshold at 0.5)
    downsampled_point_masks = downsampled_point_masks_raw > 0.5
    
    return downsampled_world_points, downsampled_normals, downsampled_base_colors, downsampled_point_masks


def rotate_array_around_center(array, angle_degrees, center=None, fill_value=0):
    """
    Rotate a 2D or 3D array around its center.
    
    Args:
        array (np.ndarray): Array to rotate, shape (H, W) or (H, W, C)
        angle_degrees (float): Rotation angle in degrees (positive = counter-clockwise)
        center (tuple): Center of rotation (cx, cy). If None, uses image center
        fill_value: Value to use for areas outside the original image
    
    Returns:
        np.ndarray: Rotated array with same shape as input
    """
    h, w = array.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    
    # Get rotation matrix for 2D rotation around center
    rotation_matrix_2d = cv2.getRotationMatrix2D(center, angle_degrees, scale=1.0)
    
    # Apply rotation based on array dimensions
    if len(array.shape) == 2:
        # 2D array (e.g., depth map, mask)
        rotated = cv2.warpAffine(array, rotation_matrix_2d, (w, h), 
                                 flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=fill_value)
    else:
        # 3D array (e.g., image, point map)
        rotated = cv2.warpAffine(array, rotation_matrix_2d, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=fill_value)
    
    return rotated


def update_extrinsics_for_image_rotation(extrinsics, angle_degrees):
    """
    Update camera extrinsics after rotating the image around its center.
    
    When we rotate an image by angle θ around its center, the camera's orientation
    relative to the world changes. Specifically, the camera rotates around its 
    viewing axis (Z-axis in camera frame).
    
    Args:
        extrinsics (np.ndarray): 4x4 camera-to-world extrinsic matrix
        angle_degrees (float): Rotation angle in degrees (positive = counter-clockwise in image)
    
    Returns:
        np.ndarray: Updated 4x4 extrinsic matrix
    """
    # Convert angle to radians (note: image rotation is counter-clockwise for positive angles)
    # In camera frame, rotating image counter-clockwise means rotating camera clockwise around Z
    # But no need to negate because the current extrinsics is Blender, where the y and z axis are negated to opencv space.
    angle_rad = np.radians(angle_degrees) 
    
    # Create rotation matrix around Z-axis in camera frame
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R_z_camera = np.array([
        [cos_a, -sin_a, 0, 0],
        [sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Convert camera-to-world extrinsics to world-to-camera
    # extrinsics is camera-to-world, so we need its inverse
    world_to_camera = np.linalg.inv(extrinsics)
    
    # Apply rotation in camera frame: new_world_to_camera = R_z * old_world_to_camera
    new_world_to_camera = R_z_camera @ world_to_camera
    
    # Convert back to camera-to-world
    new_extrinsics = np.linalg.inv(new_world_to_camera)
    
    return new_extrinsics

def randomly_expand_binary_mask(
    binary_mask,
    expand_prob=0.8,
    target_area_ratio_range=(1.3, 6.0),
    extreme_expand_prob=0.25,
    extreme_ratio_range=(6.0, 12.0),
    kernel_size_range=(3, 31),
    iteration_range=(1, 2),
    max_dilation_rounds=10,
):
    """
    Randomly expand a binary mask to simulate imperfect real-world segmentation.

    Args:
        binary_mask (np.ndarray): Input mask of shape (H, W) or squeezable to (H, W).
        expand_prob (float): Probability of applying expansion.
        target_area_ratio_range (tuple): Target expanded area ratio range relative to
            original foreground area.
        extreme_expand_prob (float): Probability of sampling a very aggressive target
            ratio from extreme_ratio_range.
        extreme_ratio_range (tuple): Target area ratio range used for aggressive cases.
        kernel_size_range (tuple): Inclusive range for random dilation kernel size
            in each dilation round.
        iteration_range (tuple): Inclusive range for random dilation iterations
            in each dilation round.
        max_dilation_rounds (int): Maximum number of randomized dilation rounds.

    Returns:
        np.ndarray: Expanded boolean mask of shape (H, W).
    """
    mask = np.asarray(binary_mask)
    if mask.ndim > 2:
        mask = np.squeeze(mask)
    mask = (mask > 0).astype(np.uint8)

    if mask.size == 0 or mask.sum() == 0:
        return mask.astype(bool)

    if np.random.rand() >= expand_prob:
        return mask.astype(bool)

    min_k, max_k = kernel_size_range
    min_iter, max_iter = iteration_range

    original_area = float(mask.sum())
    if np.random.rand() < extreme_expand_prob:
        ratio_min, ratio_max = extreme_ratio_range
    else:
        ratio_min, ratio_max = target_area_ratio_range

    target_ratio = float(np.random.uniform(ratio_min, ratio_max))
    target_area = min(mask.size, int(np.ceil(original_area * target_ratio)))

    expanded_mask = mask.copy()
    for _ in range(max_dilation_rounds):
        current_area = int(expanded_mask.sum())
        if current_area >= target_area or current_area >= mask.size:
            break

        kernel_size = int(np.random.randint(min_k, max_k + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel_shape = int(np.random.choice([
            cv2.MORPH_ELLIPSE,
            cv2.MORPH_RECT,
            cv2.MORPH_CROSS,
        ]))
        kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
        iterations = int(np.random.randint(min_iter, max_iter + 1))

        expanded_mask = cv2.dilate(expanded_mask, kernel, iterations=iterations)

    return expanded_mask.astype(bool)


def apply_rotation_augmentation(extrinsics, world_points, rotation_matrix, normals):
    """
    Apply rotation augmentation to the data.

    Args:
        extrinsics (list): List of 4x4 camera-to-world extrinsic matrices
        world_points (list): List of world point arrays
        rotation_matrix (np.ndarray): 4x4 rotation transformation matrix to apply to world
        normals (list): List of normal arrays to rotate

    Returns:
        tuple: Augmented (extrinsics, world_points, normals)
    """
    # Apply rotation to extrinsics (camera-to-world poses)
    augmented_extrinsics = []
    for extri in extrinsics:
        augmented_extri = rotation_matrix @ extri
        augmented_extrinsics.append(augmented_extri)
    
    # Apply rotation to world points
    augmented_world_points = []
    for world_pts in world_points:
        # Add homogeneous coordinate for 3D points
        if world_pts.shape[-1] == 3:
            world_pts_homo = np.concatenate([
                world_pts.reshape(-1, 3), 
                np.ones((world_pts.reshape(-1, 3).shape[0], 1))
            ], axis=1)
        else:
            world_pts_homo = world_pts.reshape(-1, 4)
        
        # Apply rotation to world points directly
        augmented_pts_homo = (rotation_matrix @ world_pts_homo.T).T
        augmented_pts = augmented_pts_homo[:, :3].reshape(world_pts.shape[:-1] + (3,))
        augmented_world_points.append(augmented_pts)

    # Apply rotation to normals list
    augmented_normals = []
    rotation_3x3 = rotation_matrix[:3, :3]
    for normal_array in normals:
        # Handle different shapes of normal arrays
        original_shape = normal_array.shape
        # Reshape to 2D for matrix multiplication
        flat_normals = normal_array.reshape(-1, 3)
        # Apply rotation (normals don't need translation)
        rotated_normals = (rotation_3x3 @ flat_normals.T).T
        # Reshape back to original shape
        rotated_normals = rotated_normals.reshape(original_shape)
        augmented_normals.append(rotated_normals)

    return augmented_extrinsics, augmented_world_points, augmented_normals

def process_bop_query_image_intrinsics(intri_opencv, query_image, binary_mask, binary_mask_copy, original_size, depth_map):
    aspect_ratio = original_size[0] / original_size[1]
    short_size = int(518 * aspect_ratio)
    small_size = 14
    # ensure the input shape is friendly to vision transformer
    if short_size % small_size != 0:
        short_size = (short_size // small_size) * small_size
    target_size = np.array([short_size, 518])

    camera_principal_point = (intri_opencv[0, 2], intri_opencv[1, 2])
    
    # the camera_principal_point is not exactly at the center of the image, we need to shift and pad the query image to offset this
    # Calculate center of the query image
    query_h, query_w = original_size
    image_center_x = query_w / 2.0
    image_center_y = query_h / 2.0
    
    # Calculate shift needed to center the principal point
    shift_x = camera_principal_point[0] - image_center_x
    shift_y = camera_principal_point[1] - image_center_y
    
    # Calculate padding needed (pad to accommodate the shift)
    pad_left = max(0, int(np.ceil(-shift_x)))
    pad_right = max(0, int(np.ceil(shift_x)))
    pad_top = max(0, int(np.ceil(-shift_y)))
    pad_bottom = max(0, int(np.ceil(shift_y)))
    
    # Pad the query image
    query_image_padded = np.pad(
        query_image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    binary_mask_padded = np.pad(
        binary_mask[..., None],
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    binary_mask_copy_padded = np.pad(
        binary_mask_copy[..., None],
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Pad the depth map
    depth_map_padded = np.pad(
        depth_map,
        ((pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant',
        constant_values=0
    )

    # Calculate crop region to get back to original size with centered principal point
    crop_x_start = pad_left + int(np.round(shift_x))
    crop_y_start = pad_top + int(np.round(shift_y))
    
    # Crop to original size
    query_image_centered = query_image_padded[
        crop_y_start:crop_y_start + query_h,
        crop_x_start:crop_x_start + query_w
    ]

    binary_mask_centered = binary_mask_padded[
        crop_y_start:crop_y_start + query_h,
        crop_x_start:crop_x_start + query_w
    ]

    binary_mask_copy_centered = binary_mask_copy_padded[
        crop_y_start:crop_y_start + query_h,
        crop_x_start:crop_x_start + query_w
    ]

    # Crop the depth map to original size
    depth_map_centered = depth_map_padded[
        crop_y_start:crop_y_start + query_h,
        crop_x_start:crop_x_start + query_w
    ]
    
    # Update the intrinsic matrix to reflect the centered principal point
    query_intrinsics = np.array([
        [intri_opencv[0, 0], 0, query_w / 2.0],
        [0, intri_opencv[1, 1], query_h / 2.0],
        [0, 0, 1]
    ])

    # Crop the query_image_centered to the foreground bounding box
    centered_h, centered_w = query_image_centered.shape[:2]
    
    # Find bounding box of the binary mask
    mask_rows = np.any(binary_mask_centered, axis=1)
    mask_cols = np.any(binary_mask_centered, axis=0)
    
    if not mask_rows.any() or not mask_cols.any():
        # If mask is empty, fall back to using the full image
        crop_y_min, crop_y_max = 0, centered_h - 1
        crop_x_min, crop_x_max = 0, centered_w - 1
    else:
        crop_y_min, crop_y_max = np.where(mask_rows)[0][[0, -1]]
        crop_x_min, crop_x_max = np.where(mask_cols)[0][[0, -1]]
    
    # Add 20% padding around the bounding box to ensure all foreground is retained
    bbox_h = crop_y_max - crop_y_min + 1
    bbox_w = crop_x_max - crop_x_min + 1
    pad_h = int(bbox_h * 0.2)
    pad_w = int(bbox_w * 0.2)
    
    crop_y_min = max(0, crop_y_min - pad_h)
    crop_y_max = min(centered_h - 1, crop_y_max + pad_h)
    crop_x_min = max(0, crop_x_min - pad_w)
    crop_x_max = min(centered_w - 1, crop_x_max + pad_w)
    
    # Get initial crop dimensions
    initial_crop_h = crop_y_max - crop_y_min + 1
    initial_crop_w = crop_x_max - crop_x_min + 1
    
    # Calculate target aspect ratio
    target_aspect_ratio = target_size[0] / target_size[1]  # height / width
    current_aspect_ratio = initial_crop_h / initial_crop_w
    
    # Adjust crop to match target aspect ratio while keeping the foreground centered
    if current_aspect_ratio > target_aspect_ratio:
        # Current crop is too tall, expand width
        new_crop_w = int(initial_crop_h / target_aspect_ratio)
        width_expansion = new_crop_w - initial_crop_w
        
        # Expand symmetrically around center
        crop_center_x = (crop_x_min + crop_x_max) // 2
        crop_x_min = max(0, crop_center_x - new_crop_w // 2)
        crop_x_max = min(centered_w - 1, crop_x_min + new_crop_w - 1)
        
        # If we hit boundary, adjust the other side
        if crop_x_max == centered_w - 1:
            crop_x_min = max(0, crop_x_max - new_crop_w + 1)
        
    else:
        # Current crop is too wide, expand height
        new_crop_h = int(initial_crop_w * target_aspect_ratio)
        height_expansion = new_crop_h - initial_crop_h
        
        # Expand symmetrically around center
        crop_center_y = (crop_y_min + crop_y_max) // 2
        crop_y_min = max(0, crop_center_y - new_crop_h // 2)
        crop_y_max = min(centered_h - 1, crop_y_min + new_crop_h - 1)
        
        # If we hit boundary, adjust the other side
        if crop_y_max == centered_h - 1:
            crop_y_min = max(0, crop_y_max - new_crop_h + 1)
    
    # Crop the image from query_image_centered
    query_image_cropped = query_image_centered[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
    crop_h, crop_w = query_image_cropped.shape[:2]

    binary_mask_cropped = binary_mask_centered[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]

    binary_mask_copy_cropped = binary_mask_copy_centered[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]

    # Crop the depth map with the same coordinates
    depth_map_cropped = depth_map_centered[crop_y_min:crop_y_max+1, crop_x_min:crop_x_max+1]
    
    # Update intrinsics: adjust principal point based on crop offset from the centered image
    query_intrinsics = np.array([
        [intri_opencv[0, 0], 0, query_intrinsics[0, 2] - crop_x_min],
        [0, intri_opencv[1, 1], query_intrinsics[1, 2] - crop_y_min],
        [0, 0, 1]
    ])
    
    # Manual resize implementation that properly handles off-center principal points
    target_h, target_w = target_size[0], target_size[1]
    
    # Calculate a single scale factor that ensures the resized image is at least as large as target in both dimensions
    scale_h = target_h / crop_h
    scale_w = target_w / crop_w
    scale = max(scale_h, scale_w)  # Use the larger scale to ensure we can fit the target size
    
    # Calculate new dimensions after uniform scaling (use ceil to ensure we're always >= target size)
    new_h = int(np.ceil(crop_h * scale))
    new_w = int(np.ceil(crop_w * scale))
    
    # Resize the image using cv2 with uniform scale
    query_image_resized = cv2.resize(query_image_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    binary_mask_resized = cv2.resize(binary_mask_cropped.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)

    binary_mask_copy_resized = cv2.resize(binary_mask_copy_cropped.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Resize the depth map with the same scale (use nearest neighbor to preserve depth values)
    depth_map_resized = cv2.resize(depth_map_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Update intrinsics for the resize operation with uniform scale
    # When resizing uniformly, both focal lengths and principal point coordinates scale by the same factor
    query_intrinsics_resized = query_intrinsics.copy()
    query_intrinsics_resized[0, 0] *= scale  # fx
    query_intrinsics_resized[1, 1] *= scale  # fy
    query_intrinsics_resized[0, 2] *= scale  # cx
    query_intrinsics_resized[1, 2] *= scale  # cy
    
    # Crop to exact target size, centered around the image center
    resized_h, resized_w = query_image_resized.shape[:2]
    
    # Calculate crop offsets to center the crop region
    start_y = (resized_h - target_h) // 2
    start_x = (resized_w - target_w) // 2
    
    # Perform center crop
    query_image_final = query_image_resized[start_y:start_y + target_h, start_x:start_x + target_w]

    binary_mask_final = binary_mask_resized[start_y:start_y + target_h, start_x:start_x + target_w]

    binary_mask_copy_final = binary_mask_copy_resized[start_y:start_y + target_h, start_x:start_x + target_w]

    # Perform center crop on depth map
    depth_map_final = depth_map_resized[start_y:start_y + target_h, start_x:start_x + target_w]
    
    # Update intrinsics for the center crop
    query_intrinsics_final = query_intrinsics_resized.copy()
    query_intrinsics_final[0, 2] -= start_x  # Adjust cx
    query_intrinsics_final[1, 2] -= start_y  # Adjust cy

    return query_image_final, query_intrinsics_final, depth_map_final, binary_mask_final, binary_mask_copy_final

class GSODataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        data_root: str = None,
        min_num_images: int = 24,
        rotation_augmentation: bool = True,
        max_rotation_angle: float = 180.0,
        sonata_downsample_stride: int = 7,
        downsample_method: str = "stride",
    ):
        """
        Initialize the GSODataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            data_root (str): Directory path to GSODataset data (rendering data with images and transforms.json).
                            Also loads from additional folders: renders3-camTrans-edited, renders3-edited, 
                            renders3-envMap, and renders3-camTrans.
            min_num_images (int): Minimum number of images per sequence.
            rotation_augmentation (bool): Whether to apply random rotation augmentation during training.
            max_rotation_angle (float): Maximum rotation angle in degrees for augmentation.
            use_sonata (bool): Whether to use SONATA point cloud processing.
            sonata_downsample_stride (int): Downsample resolution for SONATA point cloud construction.
                                               If None, uses original resolution. If specified, downsamples
                                               the point maps to this resolution before extracting points.
            downsample_method (str): Method for downsampling. Either "stride" for simple stride-based
                                   downsampling or "torch_interp" for PyTorch bilinear interpolation
                                   with align_corners=False.
        Raises:
            ValueError: If data_root is not specified.
        """
        super().__init__(common_conf=common_conf)

        # Sonata is always used as the geometry encoder.
        self.use_sonata = True
        self.sonata_transform = sonata.transform.default()
        self.sonata_downsample_stride = sonata_downsample_stride
        self.downsample_method = downsample_method
        
        # Validate downsample_method
        if self.downsample_method not in ["stride", "torch_interp"]:
            raise ValueError(f"downsample_method must be either 'stride' or 'torch_interp', got: {self.downsample_method}")
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        # Rotation augmentation settings
        self.rotation_augmentation = rotation_augmentation# and self.training  # Only during training
        self.max_rotation_angle = max_rotation_angle

        if data_root is None:
            raise ValueError("data_root must be specified.")

        # Define additional folders to load from
        self.additional_folders = [
            '../../GSO/extracted',
            # 'renders3-edited', 
            # 'renders3-envMap',
            # 'renders3-camTrans'
        ]

        # load key_to_shard.json and gso_models.json
        key_to_shard_path = osp.join(data_root, "../../GSO/key_to_shard.json")
        gso_models_path = osp.join(data_root, "../../GSO/gso_models.json")

        with open(key_to_shard_path, "r") as f:
            self.key_to_shard = json.load(f)

        with open(gso_models_path, "r") as f:
            self.gso_models = json.load(f)

        # Define depth sharing mapping
        self.depth_sharing = {
            # 'renders3-edited': 'renders3',
            # 'renders3-camTrans-edited': 'renders3-camTrans'
        }

        self.all_data_sha = os.listdir(data_root)

        # remove data where mesh.npz does not exist
        self.all_data_sha = [
            sha for sha in self.all_data_sha
            if osp.exists(osp.join(data_root, sha, "transforms.json"))
        ]

        if split == "train":
            self.data_sha = self.all_data_sha[:int(0.8 * len(self.all_data_sha))]
        elif split == "test":
            self.data_sha = self.all_data_sha[int(0.8 * len(self.all_data_sha)):]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.category_map = {}
        self.data_store = {}
        self.additional_data_store = {}  # Store data from additional folders
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"data_root is {data_root}")

        self.data_root = data_root
        total_frame_num = 0

        # Load data from main folder (renders3)
        for sha in self.data_sha:
            annotation_file = osp.join(
                self.data_root, sha, "transforms.json"
            )

            try:
                annotation = json.load(open(annotation_file, "r"))["frames"]
            except FileNotFoundError:
                logging.error(f"Annotation file not found: {annotation_file}")
                continue

            # Remove items where the image or depth file does not exist
            annotation = [
                frame for frame in annotation
                if osp.exists(osp.join(self.data_root, sha, frame["file_path"])) and
                osp.exists(osp.join(self.data_root, sha, frame["file_path"].replace(".png", "_depth.png")))
            ]

            if len(annotation) < min_num_images:
                continue
            total_frame_num += len(annotation)
            self.data_store[sha] = annotation

        # Load data from additional folders
        for folder in self.additional_folders:
            folder_path = os.path.join(data_root, folder)
            if not osp.exists(folder_path):
                logging.warning(f"Additional folder not found: {folder_path}")
                continue
                
            self.additional_data_store[folder] = {}

            for img_id, shard_id in self.key_to_shard.items():                
                gt_path = osp.join(folder_path, img_id + ".gt.json")
                with open(gt_path, "r") as f:
                    gt_data = json.load(f)
                for ind, item in enumerate(gt_data):
                    obj_id = int(item["obj_id"])
                    sha = self.gso_models[obj_id]["gso_id"]

                    if sha not in self.additional_data_store[folder]:
                        self.additional_data_store[folder][sha] = [[gt_path, ind]]
                    else:
                        self.additional_data_store[folder][sha].append([gt_path, ind])

        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: GSO Data size: {self.sequence_list_len}")
        logging.info(f"{status}: GSO Data length of training set: {len(self)}")
        
        # Log rotation augmentation status
        if self.rotation_augmentation:
            logging.info(f"{status}: Rotation augmentation enabled with max angle: {self.max_rotation_angle}°")
        else:
            logging.info(f"{status}: Rotation augmentation disabled")
        
        # Log SONATA downsampling status
        if self.use_sonata:
            if self.sonata_downsample_stride is not None:
                logging.info(f"{status}: SONATA downsampling enabled with stride: {self.sonata_downsample_stride}, method: {self.downsample_method}")
            else:
                logging.info(f"{status}: SONATA using original resolution (no downsampling)")
        else:
            logging.info(f"{status}: SONATA disabled")
        
        # Log additional folder statistics
        for folder in self.additional_folders:
            if folder in self.additional_data_store:
                folder_seqs = len(self.additional_data_store[folder])
                logging.info(f"{status}: {folder} has {folder_seqs} sequences")
            else:
                logging.info(f"{status}: {folder} not found or has no valid sequences")
        
        # Initialize augmentation pipeline lazily to ensure proper seeding in distributed training
        self.aug_pipeline = None

    def _get_aug_pipeline(self):
        """Lazy initialization of augmentation pipeline for proper seeding in distributed training."""
        if self.aug_pipeline is None:
            self.aug_pipeline = iaa.Sequential([
                # Various blur types with 30% probability
                iaa.Sometimes(0.3, iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0.5, 2.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 7)),
                    iaa.MotionBlur(k=(3, 7), angle=(-45, 45)),
                ])),
                
                # Various noise types with 30% probability
                iaa.Sometimes(0.3, iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
                    iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255)),
                    iaa.AdditivePoissonNoise(lam=(0, 20)),
                    iaa.SaltAndPepper(p=(0.01, 0.05)),
                    iaa.ImpulseNoise(p=(0.01, 0.05)),
                ])),
                
                # JPEG compression with 30% probability
                iaa.Sometimes(0.3, iaa.JpegCompression(compression=(50, 99))),
                
                # Coarse dropout with 30% probability
                iaa.Sometimes(0.3, iaa.CoarseDropout(
                    p=(0.02, 0.2),  # Probability of dropout per pixel
                    size_percent=(0.02, 0.1),  # Size of dropped rectangles
                    per_channel=0.5  # 50% chance to apply per channel
                )),
            ], random_order=True)
        return self.aug_pipeline
    
    def arrange_data(self, all_annos, all_sources, seq_name, target_image_shape):
        images = []
        base_colors = []
        normals = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for i, (anno, source_folder) in enumerate(zip(all_annos, all_sources)):
            if source_folder == 'renders3-color':
                filepath = anno["file_path"]
                current_data_root = self.data_root
                depth_data_root = self.data_root

                base_color_root = self.data_root.replace('renders3-color', 'renders3-color')
                normal_root = self.data_root.replace('renders3-color', 'renders3-normal')
                image_path = osp.join(current_data_root, seq_name, filepath)
                image = read_image_cv2(image_path)

                base_color_path = osp.join(base_color_root, seq_name, filepath)
                base_color = read_image_cv2(base_color_path)

                normal_path = osp.join(normal_root, seq_name, filepath)
                normal = read_image_cv2(normal_path)

                # Load depth map and convert to true values using min/max from annotation
                depth_path = osp.join(depth_data_root, seq_name, filepath).replace(".png", "_depth.png")
                depth_map = read_depth(depth_path, 1.0)

                # Set the value of pure white pixels to 0 which are the background
                depth_map[depth_map == 65535] = 0

                # normalize depth map to [0, 1] range
                depth_map = depth_map / 65535.0
                
                # Convert normalized depth to true depth values using annotation depth range
                depth_info = anno["depth"]
                depth_min = depth_info["min"]
                depth_max = depth_info["max"]
                # Only apply depth scaling to valid (non-zero) pixels
                valid_mask = depth_map != 0
                depth_map[valid_mask] = depth_map[valid_mask] * (depth_max - depth_min) + depth_min

                near = 0.1
                far = 100.0

                # this could be because the nvdiffrast for depth image, not use blender for depth image
                depth_map[valid_mask] = (near * far) / (far - depth_map[valid_mask] * (far - near))

                original_size = np.array(image.shape[:2])
                intri_opencv, extri_opencv = parse_camera_parameters(
                    anno,
                    image_width=original_size[1],
                    image_height=original_size[0],
                )
            else:
                gt_path, ind = anno
                with open(gt_path, "r") as f:
                    gt_data = json.load(f)[ind]

                current_data_root = self.data_root
                depth_data_root = self.data_root

                base_color_path = gt_path.replace(".gt.json", ".rgb.jpg")
                base_color = read_image_cv2(base_color_path)

                mask_anno_info_path = gt_path.replace(".gt.json", ".mask_visib.json")
                with open(mask_anno_info_path, "r") as f:
                    mask_anno_info = json.load(f)
                mask_info = mask_anno_info[str(ind)]
                binary_mask = maskUtils.decode(maskUtils.frPyObjects(
                    mask_info, mask_info["size"][0], mask_info["size"][1]
                ))

                base_color = base_color# * binary_mask[:, :, None]

                image = base_color.copy()

                normal = base_color.copy() * 0

                depth_scale = 1000.0  # depth is stored in mm, convert to meter
                depth_path = gt_path.replace(".gt.json", ".depth.png")
                depth_map = read_depth(depth_path, 1.0)
                depth_map = np.asarray(depth_map, dtype=np.float32)
                depth_map /= depth_scale

                depth_map = depth_map# * binary_mask

                original_size = np.array(image.shape[:2])

                cam_info_path = gt_path.replace(".gt.json", ".camera.json")
                with open(cam_info_path, "r") as f:
                    cam_info = json.load(f)
                intri_opencv = np.array(cam_info['cam_K']).reshape(3, 3)

                binary_mask_copy = binary_mask.copy()
                if self.training:
                    binary_mask = randomly_expand_binary_mask(
                        binary_mask,
                        expand_prob=0.95,
                        target_area_ratio_range=(2.0, 10.0),
                        extreme_expand_prob=0.5,
                        extreme_ratio_range=(10.0, 20.0),
                        kernel_size_range=(5, 51),
                        iteration_range=(1, 3),
                        max_dilation_rounds=15,
                    )

                # process bop query image
                image, intri_opencv, depth_map, binary_mask, binary_mask_copy = process_bop_query_image_intrinsics(
                    intri_opencv, image, binary_mask, binary_mask_copy, original_size, depth_map
                )
                depth_map = depth_map * binary_mask_copy
                if self.training:
                    binary_mask = randomly_expand_binary_mask(binary_mask)

                image = image * binary_mask[:, :, None]
                # depth_map = depth_map * binary_mask

                base_color = image.copy()
                original_size = np.array(image.shape[:2])

                R, T = gt_data['cam_R_m2c'], gt_data['cam_t_m2c']
                R = np.array(R).reshape(3, 3)
                T = np.array(T).reshape(3,)
                origin_intrin = intri_opencv.copy()                
                intri_opencv[0, 2], intri_opencv[1, 2] = original_size[1] // 2, original_size[0] // 2

                R, T = convert_pose_between_intrinsics(
                    origin_intrin,
                    intri_opencv,
                    np.array(R).reshape(3, 3),
                    np.array(T).reshape(3,),
                )
                extri_opencv = np.eye(4)

                flip_y_z = np.array([
                    [1,  0,  0],
                    [0,  -1,  0],
                    [0,  0,  -1]
                ])

                extri_opencv[:3, :3] = np.array(R).reshape(3, 3).T @ flip_y_z
                extri_opencv[:3, 3] = (-np.array(R).reshape(3, 3).T @ (np.array(T) * 0.001).reshape(3, 1)).squeeze()

                depth_map = depth_map * float(cam_info['depth_scale'])
                
            (
                image,
                base_color,  # base color image
                normal,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                base_color,  # base color image
                normal,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )


            # Random rotate images and point maps for data augmentation around image center
            if self.rotation_augmentation and i >= 10:
                # Generate random rotation angle
                rotation_angle = np.random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                
                # Rotate image and related arrays around image center
                image = rotate_array_around_center(image, rotation_angle, fill_value=0)
                base_color = rotate_array_around_center(base_color, rotation_angle, fill_value=0)
                normal = rotate_array_around_center(normal, rotation_angle, fill_value=0)
                depth_map = rotate_array_around_center(depth_map, rotation_angle, fill_value=0)
                
                # Rotate point maps
                world_coords_points = rotate_array_around_center(world_coords_points, rotation_angle, fill_value=0)
                cam_coords_points = rotate_array_around_center(cam_coords_points, rotation_angle, fill_value=0)
                point_mask = rotate_array_around_center(point_mask.astype(np.float32), rotation_angle, fill_value=0)
                point_mask = point_mask > 0.5  # Re-binarize mask after rotation
                
                # Update extrinsics for the image rotation (intrinsics remain unchanged)
                extri_opencv = update_extrinsics_for_image_rotation(extri_opencv, rotation_angle)

            # # save cam_coords_points
            # world_coords_points = world_coords_points.reshape(-1, 3)
            # np.savez_compressed(image_path.replace('.png', "cam_coords_points.npz"), cam_coords_points=world_coords_points)

            normal = (normal / 255.0 - 0.5) * 2.0

            # normalize normal
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = normal / (norm + 1e-8)

            images.append(image)
            base_colors.append(base_color)
            normals.append(normal)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        return images, base_colors, normals, depths, extrinsics, intrinsics, cam_points, world_points, point_masks, image_paths, original_sizes

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]


        # Determine how many images to take from additional folders (random from 1 to img_per_seq-1)
        # But only if we have additional folders with data for this sequence
        available_additional_folders = [
            folder for folder in self.additional_folders 
            if folder in self.additional_data_store and seq_name in self.additional_data_store[folder]
        ]

        if img_per_seq <= 1:
            raise NotImplementedError("This case should not happen.")

        # Random number of additional images from 1 to (img_per_seq-1)
        num_additional_total = np.random.randint(1, img_per_seq - 9) if available_additional_folders else 0
        num_main = img_per_seq - num_additional_total

        if num_main < 10:
            raise ValueError("num_main should be at least 10 to ensure some images are unmasked.")

        # Sample IDs for main folder
        if ids is None:
            random_main_ids = np.random.choice(
                len(metadata), num_main-10, replace=True
            )

            if num_main > 0:
                # Extract camera positions from transform matrices for FPS
                camera_positions = []
                for frame_data in metadata:
                    transform_matrix = np.array(frame_data['transform_matrix'])
                    # Extract camera position (translation part of transform matrix)
                    camera_position = transform_matrix[:3, 3]
                    camera_positions.append(camera_position)
                
                camera_positions = np.array(camera_positions)
                
                # Use FPS to sample diverse camera viewpoints
                main_ids = farthest_point_sampling(camera_positions, 10)

                main_ids = np.concatenate([main_ids, random_main_ids], axis=0)
            else:
                main_ids = np.array([], dtype=int)
        else:
            raise NotImplementedError("This case should not happen.")
            # If specific ids are provided, use them but limit to num_main
            main_ids = ids[:num_main] if len(ids) >= num_main else ids

        # Collect annotations from main folder and additional folders
        all_annos = []
        all_sources = []  # Track which folder each annotation comes from
        
        # Add annotations from main folder
        if len(main_ids) > 0:
            main_annos = [metadata[i] for i in main_ids]
            all_annos.extend(main_annos)
            all_sources.extend(['renders3-color'] * len(main_annos))
        
        # Add annotations from additional folders
        if num_additional_total > 0 and available_additional_folders:
            # Randomly sample from subdatasets for each additional image
            for _ in range(num_additional_total):
                # Randomly choose a subdataset
                chosen_folder = np.random.choice(available_additional_folders)
                
                # Randomly choose a frame from the chosen subdataset
                additional_metadata = self.additional_data_store[chosen_folder][seq_name]
                if not additional_metadata or len(additional_metadata) == 0:
                    raise ValueError(f"No additional metadata found in {chosen_folder} for sequence {seq_name}")

                frame_id = np.random.choice(len(additional_metadata))
                
                # Add the annotation and source
                all_annos.append(additional_metadata[frame_id])
                all_sources.append(chosen_folder)

        # Verify we have the correct total number of images
        actual_total = len(all_annos)

        assert actual_total == img_per_seq, f"Total images {actual_total} should equal expected {img_per_seq}"

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, base_colors, normals, depths, extrinsics, intrinsics, cam_points, world_points, point_masks, image_paths, original_sizes = self.arrange_data(
            all_annos, all_sources, seq_name, target_image_shape
        )

        # Validate images - check both full resolution and downsampled versions
        try_time = 0
        while True:
            try_time += 1
            needs_fallback = False
            for m, im in zip(point_masks[:10], base_colors[:10]):
                # Check full resolution image
                if np.sum(im) < 1e-5 or np.sum(m) < 1e-5:
                    needs_fallback = True
                    break
                
                # If using SONATA with downsampling, also check the downsampled version
                if self.use_sonata and self.sonata_downsample_stride is not None:
                    stride = self.sonata_downsample_stride
                    
                    if self.downsample_method == "stride":
                        downsampled_im = im[::stride, ::stride]
                        downsampled_mask = m[::stride, ::stride]
                    
                    # Check if downsampled image has sufficient content
                    if np.sum(downsampled_im) < 1e-5 or np.sum(downsampled_mask) < 1e-5:
                        needs_fallback = True
                        break
            
            if needs_fallback:
                num_additional_total = np.random.randint(1, img_per_seq - 9) if available_additional_folders else 0
                num_main = img_per_seq - num_additional_total

                random_main_ids = np.random.choice(
                    len(metadata), num_main-10, replace=True
                )

                # Extract camera positions from transform matrices for FPS
                camera_positions = []
                for frame_data in metadata:
                    transform_matrix = np.array(frame_data['transform_matrix'])
                    # Extract camera position (translation part of transform matrix)
                    camera_position = transform_matrix[:3, 3]
                    camera_positions.append(camera_position)
                
                camera_positions = np.array(camera_positions)
                
                # Use FPS to sample diverse camera viewpoints
                main_ids = farthest_point_sampling(camera_positions, 10)

                main_ids = np.concatenate([main_ids, random_main_ids], axis=0)

                # Collect annotations from main folder and additional folders
                all_annos = []
                all_sources = []  # Track which folder each annotation comes from
                
                # Add annotations from main folder
                main_annos = [metadata[i] for i in main_ids]
                all_annos.extend(main_annos)
                all_sources.extend(['renders3-color'] * len(main_annos))
                
                # Add annotations from additional folders
                if num_additional_total > 0 and available_additional_folders:
                    # Randomly sample from subdatasets for each additional image
                    for _ in range(num_additional_total):
                        # Randomly choose a subdataset
                        chosen_folder = np.random.choice(available_additional_folders)
                        
                        # Randomly choose a frame from the chosen subdataset
                        additional_metadata = self.additional_data_store[chosen_folder][seq_name]
                        if not additional_metadata or len(additional_metadata) == 0:
                            raise ValueError(f"No additional metadata found in {chosen_folder} for sequence {seq_name}")

                        frame_id = np.random.choice(len(additional_metadata))
                        
                        # Add the annotation and source
                        all_annos.append(additional_metadata[frame_id])
                        all_sources.append(chosen_folder)
                
                images, base_colors, normals, depths, extrinsics, intrinsics, cam_points, world_points, point_masks, image_paths, original_sizes = self.arrange_data(
                    all_annos, all_sources, seq_name, target_image_shape
                )
            else:
                break

            if try_time >= 20:
                logging.warning(f"Too many fallback attempts for sequence {seq_name}, moving to a new sequence.")
            
                seq_index = random.randint(0, self.sequence_list_len - 1)
                seq_name = self.sequence_list[seq_index]

                metadata = self.data_store[seq_name]

                available_additional_folders = [
                    folder for folder in self.additional_folders 
                    if folder in self.additional_data_store and seq_name in self.additional_data_store[folder]
                ]
                try_time = 0 # reset

        set_name = "GSO"

        # Apply rotation augmentation if enabled
        if self.rotation_augmentation:
            rotation_matrix = generate_random_rotation(self.max_rotation_angle)
            extrinsics, world_points, normals = apply_rotation_augmentation(
                extrinsics, world_points, rotation_matrix, normals
            )

        # Generate mask IDs for all images
        total_images = len(all_annos)
        mask_ids = np.zeros(total_images, dtype=np.int32)
        
        # For the main folder images, apply random masking
        main_images_count = num_main

        mask_ids[:10] = 1

        if mask_ids.all():
            raise ValueError("All images are masked, which should not happen.")

        if self.use_sonata:
            point_sonata = dict()
            
            valid_coords = []
            valid_normals = []
            valid_colors = []
            
            for i in range(10):
                # Get arrays for this image
                current_world_pts = world_points[i]  # Shape: (H, W, 3)
                current_normals = normals[i]         # Shape: (H, W, 3)
                current_colors = base_colors[i]      # Shape: (H, W, 3)
                current_mask = point_masks[i]        # Shape: (H, W)
                
                # Apply downsampling if specified
                if self.sonata_downsample_stride is not None:
                    if self.downsample_method == "torch_interp":
                        current_world_pts, current_normals, current_colors, current_mask = downsample_point_arrays_torch_interp(
                            current_world_pts, current_normals, current_colors, current_mask, 
                            self.sonata_downsample_stride
                        )
                    else:  # Default to stride method
                        current_world_pts, current_normals, current_colors, current_mask = downsample_point_arrays(
                            current_world_pts, current_normals, current_colors, current_mask, 
                            self.sonata_downsample_stride
                        )
                
                # Extract valid points using the mask
                valid_world_pts = current_world_pts[current_mask]  # Shape: (N_valid, 3)
                valid_normal_pts = current_normals[current_mask]   # Shape: (N_valid, 3)
                valid_color_pts = current_colors[current_mask]     # Shape: (N_valid, 3)
                
                valid_coords.append(valid_world_pts)
                valid_normals.append(valid_normal_pts)
                valid_colors.append(valid_color_pts)
            
            # Check if we have any valid points
            if len(valid_coords) == 0 or all(len(v) == 0 for v in valid_coords):
                print('All annotations:', all_annos)
                print('All sources:', all_sources)

                # raise error and print out the image_path
                raise ValueError(f"No valid SONATA points found in the first 10 images of sequence {seq_name}. Image paths: {image_paths[:10]}")
            else:
                point_sonata['coord'] = np.concatenate(valid_coords, axis=0)
                point_sonata['normal'] = np.concatenate(valid_normals, axis=0)
                # # normal here is 0 - 255, need to convert to 0 - 1, and then to -1 to 1
                # point_sonata['normal'] = (point_sonata['normal'] / 255.0 - 0.5) * 2.0
                point_sonata['color'] = np.concatenate(valid_colors, axis=0)
                # color here is 0 - 255, need to convert to 0 - 1
                point_sonata['color'] = point_sonata['color'] / 255.0

                initial_sonata_num = point_sonata['coord'].shape[0]

                try:
                    point_sonata = self.sonata_transform(point_sonata)
                except Exception as e:
                    print('All annotations:', all_annos)
                    print('All sources:', all_sources)

                    # raise error and print out the image_path
                    raise ValueError(f"Second part: No valid SONATA points found in the first 10 images of sequence {seq_name}. Image paths: {image_paths[:10]}")

        # Mask out images, base_colors, depths, cam_points, world_points, point_masks for images of num_additional_total by using Eclipse or Rectangle or Freeform mask. The possibility of this masking is 20%. Ensure the masked area covers at least 10% and at most 50% of the foreground area. You can leverage the point_masks to determine the foreground area.
        
        # Apply masking to additional images with 50% probability
        if np.random.rand() < 0.5:
            # Select indices of additional images (after main images)
            additional_indices = list(range(10, img_per_seq))
            
            for idx in additional_indices:
                # Get current image shape
                img_height, img_width = images[idx].shape[:2]
                
                # Use point_mask to determine foreground area
                foreground_mask = point_masks[idx]
                foreground_area = np.sum(foreground_mask)
                total_area = img_height * img_width
                
                if foreground_area == 0:
                    continue  # Skip if no foreground
                
                # Try multiple times to generate a valid mask
                max_attempts = 10
                valid_mask_found = False
                
                for attempt in range(max_attempts):
                    # Randomly choose mask type: 0=rectangle, 1=ellipse, 2=free_form
                    mask_type = np.random.randint(0, 3)
                    
                    if mask_type == 2:  # Free form mask
                        # Use the smaller dimension for resolution to ensure mask fits
                        resolution = min(img_height, img_width)
                        generated_mask = free_form_mask(resolution)
                        # Resize to match image dimensions if needed
                        if resolution != img_height or resolution != img_width:
                            generated_mask = generated_mask.resize((img_width, img_height), Image.LANCZOS)
                    else:
                        # Use random_mask for rectangle/ellipse
                        generated_mask = random_mask((img_width, img_height), ratio=1, mask_full_image=False)
                    
                    # Convert mask to numpy array
                    mask_array = prepare_mask_and_masked_image(generated_mask)
                    mask_array = mask_array.squeeze().numpy()  # Remove batch dimension and convert to numpy
                    
                    # Calculate overlap between generated mask and foreground
                    masked_foreground_area = np.sum(mask_array * foreground_mask.astype(np.float32))
                    foreground_mask_ratio = masked_foreground_area / foreground_area
                    
                    # Check if the mask covers 10-50% of foreground area
                    if 0.1 <= foreground_mask_ratio <= 0.5:
                        valid_mask_found = True
                        break
                
                if valid_mask_found:
                    # Apply mask to all relevant data
                    mask_3d = mask_array[..., np.newaxis]  # Add channel dimension for broadcasting
                    
                    # Apply mask to images (set masked areas to 0)
                    masked_value = 0.0  # Black for images
                    images[idx] = images[idx] * (1 - mask_3d) + masked_value * mask_3d

                    # save images[idx] for visualization
                    # Image.fromarray((images[idx]).astype(np.uint8)).save(f"debug_masked_image_{idx}.png")
                    
                    # Apply mask to depths (set masked areas to 0)
                    if len(depths) > idx and depths[idx] is not None:
                        depths[idx] = depths[idx] * (1 - mask_array)
                    
                    # Apply mask to cam_points and world_points (set masked areas to 0)
                    if len(cam_points) > idx and cam_points[idx] is not None:
                        cam_points[idx] = cam_points[idx] * (1 - mask_3d)
                    
                    if len(world_points) > idx and world_points[idx] is not None:
                        world_points[idx] = world_points[idx] * (1 - mask_3d)
                    
                    # Update point_masks (masked areas become invalid)
                    if len(point_masks) > idx and point_masks[idx] is not None:
                        point_masks[idx] = point_masks[idx] * (1 - mask_array.astype(bool))
        
        # apply imgaug augmentation to additional images. Considering different kinds of blur, kinds of noise, jpeg compression, coarse dropout, each with a probability of 0.3 and a random scale set. 
        additional_indices = list(range(10, img_per_seq))
            
        for idx in additional_indices:
            
            # Apply augmentation to image
            # imgaug expects uint8 images
            img_uint8 = images[idx].astype(np.uint8)

            # if all pixels are black, skip augmentation and set extrinsics to identity
            if np.sum(point_masks[idx]) < 1e-5:
                extrinsics[idx] = np.eye(4, dtype=extrinsics[idx].dtype)
                continue

            # get foreground mask from image where pixels are not black
            foreground_mask = np.any(images[idx] > 5, axis=-1).astype(np.float32)
            
            # Apply the augmentation to images
            augmented_img = self._get_aug_pipeline()(image=img_uint8)

            # Blend augmented image with original using foreground mask to avoid altering black background too much
            foreground_mask_3d = foreground_mask[..., np.newaxis]
            augmented_img = images[idx] * (1 - foreground_mask_3d) + augmented_img * foreground_mask_3d

            images[idx] = augmented_img.astype(np.float32)

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": np.array(list(range(total_images))),
            "camera_mask": mask_ids,
            "frame_num": len(extrinsics),
            "images": images,
            "base_colors": base_colors,
            "depths": depths if depths and depths[0] is not None else [],
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points if cam_points and cam_points[0] is not None else [],
            "world_points": world_points if world_points and world_points[0] is not None else [],
            "point_masks": point_masks if point_masks and point_masks[0] is not None else [],
            "original_sizes": original_sizes,
            "point_sonata": point_sonata,
            "initial_sonata_num": initial_sonata_num,
        }
        return batch


class ShapeNetDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        data_root: str = None,
        min_num_images: int = 24,
        rotation_augmentation: bool = True,
        max_rotation_angle: float = 180.0,
        sonata_downsample_stride: int = 7,
        downsample_method: str = "stride",
    ):
        """
        Initialize the ShapeNetDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            data_root (str): Directory path to ShapeNetDataset data (rendering data with images and transforms.json).
                            Also loads from additional folders: renders3-camTrans-edited, renders3-edited, 
                            renders3-envMap, and renders3-camTrans.
            min_num_images (int): Minimum number of images per sequence.
            rotation_augmentation (bool): Whether to apply random rotation augmentation during training.
            max_rotation_angle (float): Maximum rotation angle in degrees for augmentation.
            use_sonata (bool): Whether to use SONATA point cloud processing.
            sonata_downsample_stride (int): Downsample resolution for SONATA point cloud construction.
                                               If None, uses original resolution. If specified, downsamples
                                               the point maps to this resolution before extracting points.
            downsample_method (str): Method for downsampling. Either "stride" for simple stride-based
                                   downsampling or "torch_interp" for PyTorch bilinear interpolation
                                   with align_corners=False.
        Raises:
            ValueError: If data_root is not specified.
        """
        super().__init__(common_conf=common_conf)

        # Sonata is always used as the geometry encoder.
        self.use_sonata = True
        self.sonata_transform = sonata.transform.default()
        self.sonata_downsample_stride = sonata_downsample_stride
        self.downsample_method = downsample_method
        
        # Validate downsample_method
        if self.downsample_method not in ["stride", "torch_interp"]:
            raise ValueError(f"downsample_method must be either 'stride' or 'torch_interp', got: {self.downsample_method}")
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.load_depth = common_conf.load_depth
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        # Rotation augmentation settings
        self.rotation_augmentation = rotation_augmentation# and self.training  # Only during training
        self.max_rotation_angle = max_rotation_angle

        if data_root is None:
            raise ValueError("data_root must be specified.")

        # Define additional folders to load from
        self.additional_folders = [
            '../../ShapeNet/extracted',
            # 'renders3-edited', 
            # 'renders3-envMap',
            # 'renders3-camTrans'
        ]

        # load key_to_shard.json and shapenet_models.json
        key_to_shard_path = osp.join(data_root, "../../ShapeNet/key_to_shard.json")
        shapenet_models_path = osp.join(data_root, "../../ShapeNet/shapenet_models.json")

        with open(key_to_shard_path, "r") as f:
            self.key_to_shard = json.load(f)

        with open(shapenet_models_path, "r") as f:
            self.shapenet_models = json.load(f)

        # Define depth sharing mapping
        self.depth_sharing = {
            # 'renders3-edited': 'renders3',
            # 'renders3-camTrans-edited': 'renders3-camTrans'
        }

        # remove data where mesh.npz does not exist
        self.all_data_sha = []
        for sha in os.listdir(data_root):
            for subsha in os.listdir(osp.join(data_root, sha)):
                if osp.exists(osp.join(data_root, sha, subsha, "transforms.json")):
                    self.all_data_sha.append(sha + "/" + subsha)

        if split == "train":
            self.data_sha = self.all_data_sha[:int(0.8 * len(self.all_data_sha))]
        elif split == "test":
            self.data_sha = self.all_data_sha[int(0.8 * len(self.all_data_sha)):]
        else:
            raise ValueError(f"Invalid split: {split}")

        self.category_map = {}
        self.data_store = {}
        self.additional_data_store = {}  # Store data from additional folders
        self.seqlen = None
        self.min_num_images = min_num_images

        logging.info(f"data_root is {data_root}")

        self.data_root = data_root
        total_frame_num = 0

        # Load data from main folder (renders3)
        for sha in self.data_sha:
            annotation_file = osp.join(
                self.data_root, sha, "transforms.json"
            )

            try:
                annotation = json.load(open(annotation_file, "r"))["frames"]
            except FileNotFoundError:
                logging.error(f"Annotation file not found: {annotation_file}")
                continue

            # Remove items where the image or depth file does not exist
            annotation = [
                frame for frame in annotation
                if osp.exists(osp.join(self.data_root, sha, frame["file_path"])) and
                osp.exists(osp.join(self.data_root, sha, frame["file_path"].replace(".png", "_depth.png")))
            ]

            if len(annotation) < min_num_images:
                continue
            total_frame_num += len(annotation)
            self.data_store[sha] = annotation

        # Load data from additional folders
        for folder in self.additional_folders:
            folder_path = os.path.join(data_root, folder)
            if not osp.exists(folder_path):
                logging.warning(f"Additional folder not found: {folder_path}")
                continue
                
            self.additional_data_store[folder] = {}

            for img_id, shard_id in self.key_to_shard.items():                
                gt_path = osp.join(folder_path, img_id + ".gt.json")
                with open(gt_path, "r") as f:
                    gt_data = json.load(f)
                for ind, item in enumerate(gt_data):
                    obj_id = int(item["obj_id"])
                    sha1, sha2 = self.shapenet_models[obj_id]["shapenet_synset_id"], self.shapenet_models[obj_id]["shapenet_source_id"]
                    sha = sha1 + "/" + sha2

                    if sha not in self.additional_data_store[folder]:
                        self.additional_data_store[folder][sha] = [[gt_path, ind]]
                    else:
                        self.additional_data_store[folder][sha].append([gt_path, ind])
    
        self.sequence_list = list(self.data_store.keys())
        self.sequence_list_len = len(self.sequence_list)

        status = "Training" if self.training else "Test"
        logging.info(f"{status}: ShapeNet Data size: {self.sequence_list_len}")
        logging.info(f"{status}: ShapeNet Data length of training set: {len(self)}")
        
        # Log rotation augmentation status
        if self.rotation_augmentation:
            logging.info(f"{status}: Rotation augmentation enabled with max angle: {self.max_rotation_angle}°")
        else:
            logging.info(f"{status}: Rotation augmentation disabled")
        
        # Log SONATA downsampling status
        if self.use_sonata:
            if self.sonata_downsample_stride is not None:
                logging.info(f"{status}: SONATA downsampling enabled with stride: {self.sonata_downsample_stride}, method: {self.downsample_method}")
            else:
                logging.info(f"{status}: SONATA using original resolution (no downsampling)")
        else:
            logging.info(f"{status}: SONATA disabled")
        
        # Log additional folder statistics
        for folder in self.additional_folders:
            if folder in self.additional_data_store:
                folder_seqs = len(self.additional_data_store[folder])
                logging.info(f"{status}: {folder} has {folder_seqs} sequences")
            else:
                logging.info(f"{status}: {folder} not found or has no valid sequences")
        
        # Initialize augmentation pipeline lazily to ensure proper seeding in distributed training
        self.aug_pipeline = None

    def _get_aug_pipeline(self):
        """Lazy initialization of augmentation pipeline for proper seeding in distributed training."""
        if self.aug_pipeline is None:
            self.aug_pipeline = iaa.Sequential([
                # Various blur types with 30% probability
                iaa.Sometimes(0.3, iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0.5, 2.0)),
                    iaa.AverageBlur(k=(2, 7)),
                    iaa.MedianBlur(k=(3, 7)),
                    iaa.MotionBlur(k=(3, 7), angle=(-45, 45)),
                ])),
                
                # Various noise types with 30% probability
                iaa.Sometimes(0.3, iaa.OneOf([
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),
                    iaa.AdditiveLaplaceNoise(scale=(0, 0.05*255)),
                    iaa.AdditivePoissonNoise(lam=(0, 20)),
                    iaa.SaltAndPepper(p=(0.01, 0.05)),
                    iaa.ImpulseNoise(p=(0.01, 0.05)),
                ])),
                
                # JPEG compression with 30% probability
                iaa.Sometimes(0.3, iaa.JpegCompression(compression=(50, 99))),
                
                # Coarse dropout with 30% probability
                iaa.Sometimes(0.3, iaa.CoarseDropout(
                    p=(0.02, 0.2),  # Probability of dropout per pixel
                    size_percent=(0.02, 0.1),  # Size of dropped rectangles
                    per_channel=0.5  # 50% chance to apply per channel
                )),
            ], random_order=True)
        return self.aug_pipeline
    
    def arrange_data(self, all_annos, all_sources, seq_name, target_image_shape):
        images = []
        base_colors = []
        normals = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for i, (anno, source_folder) in enumerate(zip(all_annos, all_sources)):
            if source_folder == 'renders3-color':
                filepath = anno["file_path"]
                current_data_root = self.data_root
                depth_data_root = self.data_root

                base_color_root = self.data_root.replace('renders3-color', 'renders3-color')
                normal_root = self.data_root.replace('renders3-color', 'renders3-normal')
                image_path = osp.join(current_data_root, seq_name, filepath)
                image = read_image_cv2(image_path)

                base_color_path = osp.join(base_color_root, seq_name, filepath)
                base_color = read_image_cv2(base_color_path)

                normal_path = osp.join(normal_root, seq_name, filepath)
                normal = read_image_cv2(normal_path)

                # Load depth map and convert to true values using min/max from annotation
                depth_path = osp.join(depth_data_root, seq_name, filepath).replace(".png", "_depth.png")
                depth_map = read_depth(depth_path, 1.0)

                # Set the value of pure white pixels to 0 which are the background
                depth_map[depth_map == 65535] = 0

                # normalize depth map to [0, 1] range
                depth_map = depth_map / 65535.0
                
                # Convert normalized depth to true depth values using annotation depth range
                depth_info = anno["depth"]
                depth_min = depth_info["min"]
                depth_max = depth_info["max"]
                # Only apply depth scaling to valid (non-zero) pixels
                valid_mask = depth_map != 0
                depth_map[valid_mask] = depth_map[valid_mask] * (depth_max - depth_min) + depth_min

                near = 0.1
                far = 100.0

                depth_map[valid_mask] = (near * far) / (far - depth_map[valid_mask] * (far - near))

                original_size = np.array(image.shape[:2])
                intri_opencv, extri_opencv = parse_camera_parameters(
                    anno,
                    image_width=original_size[1],
                    image_height=original_size[0],
                )
            else:
                gt_path, ind = anno
                with open(gt_path, "r") as f:
                    gt_data = json.load(f)[ind]

                current_data_root = self.data_root
                depth_data_root = self.data_root

                base_color_path = gt_path.replace(".gt.json", ".rgb.jpg")
                base_color = read_image_cv2(base_color_path)

                mask_anno_info_path = gt_path.replace(".gt.json", ".mask_visib.json")
                with open(mask_anno_info_path, "r") as f:
                    mask_anno_info = json.load(f)
                mask_info = mask_anno_info[str(ind)]
                binary_mask = maskUtils.decode(maskUtils.frPyObjects(
                    mask_info, mask_info["size"][0], mask_info["size"][1]
                ))

                base_color = base_color# * binary_mask[:, :, None]

                image = base_color.copy()

                normal = base_color.copy() * 0

                depth_scale = 1000.0  # depth is stored in mm, convert to meter
                depth_path = gt_path.replace(".gt.json", ".depth.png")
                depth_map = read_depth(depth_path, 1.0)
                depth_map = np.asarray(depth_map, dtype=np.float32)
                depth_map /= depth_scale

                depth_map = depth_map# * binary_mask

                original_size = np.array(image.shape[:2])

                cam_info_path = gt_path.replace(".gt.json", ".camera.json")
                with open(cam_info_path, "r") as f:
                    cam_info = json.load(f)
                intri_opencv = np.array(cam_info['cam_K']).reshape(3, 3)

                binary_mask_copy = binary_mask.copy()
                if self.training:
                    binary_mask = randomly_expand_binary_mask(
                        binary_mask,
                        expand_prob=0.95,
                        target_area_ratio_range=(2.0, 10.0),
                        extreme_expand_prob=0.5,
                        extreme_ratio_range=(10.0, 20.0),
                        kernel_size_range=(5, 51),
                        iteration_range=(1, 3),
                        max_dilation_rounds=15,
                    )

                # process bop query image
                image, intri_opencv, depth_map, binary_mask, binary_mask_copy = process_bop_query_image_intrinsics(
                    intri_opencv, image, binary_mask, binary_mask_copy, original_size, depth_map
                )
                depth_map = depth_map * binary_mask_copy
                if self.training:
                    binary_mask = randomly_expand_binary_mask(binary_mask)

                image = image * binary_mask[:, :, None]
                # depth_map = depth_map * binary_mask

                base_color = image.copy()
                original_size = np.array(image.shape[:2])

                R, T = gt_data['cam_R_m2c'], gt_data['cam_t_m2c']
                R = np.array(R).reshape(3, 3)
                T = np.array(T).reshape(3,)
                origin_intrin = intri_opencv.copy()                
                intri_opencv[0, 2], intri_opencv[1, 2] = original_size[1] // 2, original_size[0] // 2

                R, T = convert_pose_between_intrinsics(
                    origin_intrin,
                    intri_opencv,
                    np.array(R).reshape(3, 3),
                    np.array(T).reshape(3,),
                )
                extri_opencv = np.eye(4)

                flip_y_z = np.array([
                    [1,  0,  0],
                    [0,  -1,  0],
                    [0,  0,  -1]
                ])

                extri_opencv[:3, :3] = np.array(R).reshape(3, 3).T @ flip_y_z
                extri_opencv[:3, 3] = (-np.array(R).reshape(3, 3).T @ (np.array(T) * 0.001).reshape(3, 1)).squeeze()

                depth_map = depth_map * float(cam_info['depth_scale'])
                
            (
                image,
                base_color,  # base color image
                normal,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                base_color,  # base color image
                normal,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=filepath,
            )


            # Random rotate images and point maps for data augmentation around image center
            if self.rotation_augmentation and i >= 10:
                # Generate random rotation angle
                rotation_angle = np.random.uniform(-self.max_rotation_angle, self.max_rotation_angle)
                
                # Rotate image and related arrays around image center
                image = rotate_array_around_center(image, rotation_angle, fill_value=0)
                base_color = rotate_array_around_center(base_color, rotation_angle, fill_value=0)
                normal = rotate_array_around_center(normal, rotation_angle, fill_value=0)
                depth_map = rotate_array_around_center(depth_map, rotation_angle, fill_value=0)
                
                # Rotate point maps
                world_coords_points = rotate_array_around_center(world_coords_points, rotation_angle, fill_value=0)
                cam_coords_points = rotate_array_around_center(cam_coords_points, rotation_angle, fill_value=0)
                point_mask = rotate_array_around_center(point_mask.astype(np.float32), rotation_angle, fill_value=0)
                point_mask = point_mask > 0.5  # Re-binarize mask after rotation
                
                # Update extrinsics for the image rotation (intrinsics remain unchanged)
                extri_opencv = update_extrinsics_for_image_rotation(extri_opencv, rotation_angle)

            # # save cam_coords_points
            # world_coords_points = world_coords_points.reshape(-1, 3)
            # np.savez_compressed(image_path.replace('.png', "cam_coords_points.npz"), cam_coords_points=world_coords_points)

            normal = (normal / 255.0 - 0.5) * 2.0

            # normalize normal
            norm = np.linalg.norm(normal, axis=-1, keepdims=True)
            normal = normal / (norm + 1e-8)

            images.append(image)
            base_colors.append(base_color)
            normals.append(normal)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            image_paths.append(image_path)
            original_sizes.append(original_size)

        return images, base_colors, normals, depths, extrinsics, intrinsics, cam_points, world_points, point_masks, image_paths, original_sizes

    def get_data(
        self,
        seq_index: int = None,
        img_per_seq: int = None,
        seq_name: str = None,
        ids: list = None,
        aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random:
            seq_index = random.randint(0, self.sequence_list_len - 1)
            
        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        metadata = self.data_store[seq_name]


        # Determine how many images to take from additional folders (random from 1 to img_per_seq-1)
        # But only if we have additional folders with data for this sequence
        available_additional_folders = [
            folder for folder in self.additional_folders 
            if folder in self.additional_data_store and seq_name in self.additional_data_store[folder]
        ]

        if img_per_seq <= 1:
            raise NotImplementedError("This case should not happen.")

        # Random number of additional images from 1 to (img_per_seq-1)
        num_additional_total = np.random.randint(1, img_per_seq - 9) if available_additional_folders else 0
        num_main = img_per_seq - num_additional_total

        if num_main < 10:
            raise ValueError("num_main should be at least 10 to ensure some images are unmasked.")

        # Sample IDs for main folder
        if ids is None:
            random_main_ids = np.random.choice(
                len(metadata), num_main-10, replace=True
            )

            if num_main > 0:
                # Extract camera positions from transform matrices for FPS
                camera_positions = []
                for frame_data in metadata:
                    transform_matrix = np.array(frame_data['transform_matrix'])
                    # Extract camera position (translation part of transform matrix)
                    camera_position = transform_matrix[:3, 3]
                    camera_positions.append(camera_position)
                
                camera_positions = np.array(camera_positions)
                
                # Use FPS to sample diverse camera viewpoints
                main_ids = farthest_point_sampling(camera_positions, 10)

                main_ids = np.concatenate([main_ids, random_main_ids], axis=0)
            else:
                main_ids = np.array([], dtype=int)
        else:
            raise NotImplementedError("This case should not happen.")
            # If specific ids are provided, use them but limit to num_main
            main_ids = ids[:num_main] if len(ids) >= num_main else ids

        # Collect annotations from main folder and additional folders
        all_annos = []
        all_sources = []  # Track which folder each annotation comes from
        
        # Add annotations from main folder
        if len(main_ids) > 0:
            main_annos = [metadata[i] for i in main_ids]
            all_annos.extend(main_annos)
            all_sources.extend(['renders3-color'] * len(main_annos))
        
        # Add annotations from additional folders
        if num_additional_total > 0 and available_additional_folders:
            # Randomly sample from subdatasets for each additional image
            for _ in range(num_additional_total):
                # Randomly choose a subdataset
                chosen_folder = np.random.choice(available_additional_folders)
                
                # Randomly choose a frame from the chosen subdataset
                additional_metadata = self.additional_data_store[chosen_folder][seq_name]
                if not additional_metadata or len(additional_metadata) == 0:
                    raise ValueError(f"No additional metadata found in {chosen_folder} for sequence {seq_name}")

                frame_id = np.random.choice(len(additional_metadata))
                
                # Add the annotation and source
                all_annos.append(additional_metadata[frame_id])
                all_sources.append(chosen_folder)

        # Verify we have the correct total number of images
        actual_total = len(all_annos)

        assert actual_total == img_per_seq, f"Total images {actual_total} should equal expected {img_per_seq}"

        target_image_shape = self.get_target_shape(aspect_ratio)

        images, base_colors, normals, depths, extrinsics, intrinsics, cam_points, world_points, point_masks, image_paths, original_sizes = self.arrange_data(
            all_annos, all_sources, seq_name, target_image_shape
        )

        # Validate images - check both full resolution and downsampled versions
        try_time = 0
        while True:
            try_time += 1
            needs_fallback = False
            for m, im in zip(point_masks[:10], base_colors[:10]):
                # Check full resolution image
                if np.sum(im) < 1e-5 or np.sum(m) < 1e-5:
                    needs_fallback = True
                    break
                
                # If using SONATA with downsampling, also check the downsampled version
                if self.use_sonata and self.sonata_downsample_stride is not None:
                    stride = self.sonata_downsample_stride
                    
                    if self.downsample_method == "stride":
                        downsampled_im = im[::stride, ::stride]
                        downsampled_mask = m[::stride, ::stride]
                    
                    # Check if downsampled image has sufficient content
                    if np.sum(downsampled_im) < 1e-5 or np.sum(downsampled_mask) < 1e-5:
                        needs_fallback = True
                        break
            
            if needs_fallback:
                num_additional_total = np.random.randint(1, img_per_seq - 9) if available_additional_folders else 0
                num_main = img_per_seq - num_additional_total

                random_main_ids = np.random.choice(
                    len(metadata), num_main-10, replace=True
                )

                # Extract camera positions from transform matrices for FPS
                camera_positions = []
                for frame_data in metadata:
                    transform_matrix = np.array(frame_data['transform_matrix'])
                    # Extract camera position (translation part of transform matrix)
                    camera_position = transform_matrix[:3, 3]
                    camera_positions.append(camera_position)
                
                camera_positions = np.array(camera_positions)
                
                # Use FPS to sample diverse camera viewpoints
                main_ids = farthest_point_sampling(camera_positions, 10)

                main_ids = np.concatenate([main_ids, random_main_ids], axis=0)

                # Collect annotations from main folder and additional folders
                all_annos = []
                all_sources = []  # Track which folder each annotation comes from
                
                # Add annotations from main folder
                main_annos = [metadata[i] for i in main_ids]
                all_annos.extend(main_annos)
                all_sources.extend(['renders3-color'] * len(main_annos))
                
                # Add annotations from additional folders
                if num_additional_total > 0 and available_additional_folders:
                    # Randomly sample from subdatasets for each additional image
                    for _ in range(num_additional_total):
                        # Randomly choose a subdataset
                        chosen_folder = np.random.choice(available_additional_folders)
                        
                        # Randomly choose a frame from the chosen subdataset
                        additional_metadata = self.additional_data_store[chosen_folder][seq_name]
                        if not additional_metadata or len(additional_metadata) == 0:
                            raise ValueError(f"No additional metadata found in {chosen_folder} for sequence {seq_name}")

                        frame_id = np.random.choice(len(additional_metadata))
                        
                        # Add the annotation and source
                        all_annos.append(additional_metadata[frame_id])
                        all_sources.append(chosen_folder)
                
                images, base_colors, normals, depths, extrinsics, intrinsics, cam_points, world_points, point_masks, image_paths, original_sizes = self.arrange_data(
                    all_annos, all_sources, seq_name, target_image_shape
                )
            else:
                break

            if try_time >= 20:
                logging.warning(f"Too many fallback attempts for sequence {seq_name}, moving to a new sequence.")
            
                seq_index = random.randint(0, self.sequence_list_len - 1)
                seq_name = self.sequence_list[seq_index]

                metadata = self.data_store[seq_name]

                available_additional_folders = [
                    folder for folder in self.additional_folders 
                    if folder in self.additional_data_store and seq_name in self.additional_data_store[folder]
                ]
                try_time = 0 # reset

        set_name = "ShapeNet"

        # Apply rotation augmentation if enabled
        if self.rotation_augmentation:
            rotation_matrix = generate_random_rotation(self.max_rotation_angle)
            extrinsics, world_points, normals = apply_rotation_augmentation(
                extrinsics, world_points, rotation_matrix, normals
            )

        # Generate mask IDs for all images
        total_images = len(all_annos)
        mask_ids = np.zeros(total_images, dtype=np.int32)
        
        # For the main folder images, apply random masking
        main_images_count = num_main

        mask_ids[:10] = 1

        if mask_ids.all():
            raise ValueError("All images are masked, which should not happen.")

        if self.use_sonata:
            point_sonata = dict()
            
            valid_coords = []
            valid_normals = []
            valid_colors = []
            
            for i in range(10):
                # Get arrays for this image
                current_world_pts = world_points[i]  # Shape: (H, W, 3)
                current_normals = normals[i]         # Shape: (H, W, 3)
                current_colors = base_colors[i]      # Shape: (H, W, 3)
                current_mask = point_masks[i]        # Shape: (H, W)
                
                # Apply downsampling if specified
                if self.sonata_downsample_stride is not None:
                    if self.downsample_method == "torch_interp":
                        current_world_pts, current_normals, current_colors, current_mask = downsample_point_arrays_torch_interp(
                            current_world_pts, current_normals, current_colors, current_mask, 
                            self.sonata_downsample_stride
                        )
                    else:  # Default to stride method
                        current_world_pts, current_normals, current_colors, current_mask = downsample_point_arrays(
                            current_world_pts, current_normals, current_colors, current_mask, 
                            self.sonata_downsample_stride
                        )
                
                # Extract valid points using the mask
                valid_world_pts = current_world_pts[current_mask]  # Shape: (N_valid, 3)
                valid_normal_pts = current_normals[current_mask]   # Shape: (N_valid, 3)
                valid_color_pts = current_colors[current_mask]     # Shape: (N_valid, 3)
                
                valid_coords.append(valid_world_pts)
                valid_normals.append(valid_normal_pts)
                valid_colors.append(valid_color_pts)
            
            # Check if we have any valid points
            if len(valid_coords) == 0 or all(len(v) == 0 for v in valid_coords):
                print('All annotations:', all_annos)
                print('All sources:', all_sources)

                # raise error and print out the image_path
                raise ValueError(f"No valid SONATA points found in the first 10 images of sequence {seq_name}. Image paths: {image_paths[:10]}")
            else:
                point_sonata['coord'] = np.concatenate(valid_coords, axis=0)
                point_sonata['normal'] = np.concatenate(valid_normals, axis=0)
                # # normal here is 0 - 255, need to convert to 0 - 1, and then to -1 to 1
                # point_sonata['normal'] = (point_sonata['normal'] / 255.0 - 0.5) * 2.0
                point_sonata['color'] = np.concatenate(valid_colors, axis=0)
                # color here is 0 - 255, need to convert to 0 - 1
                point_sonata['color'] = point_sonata['color'] / 255.0

                initial_sonata_num = point_sonata['coord'].shape[0]

                try:
                    point_sonata = self.sonata_transform(point_sonata)
                except Exception as e:
                    print('All annotations:', all_annos)
                    print('All sources:', all_sources)

                    # raise error and print out the image_path
                    raise ValueError(f"Second part: No valid SONATA points found in the first 10 images of sequence {seq_name}. Image paths: {image_paths[:10]}")

        # Mask out images, base_colors, depths, cam_points, world_points, point_masks for images of num_additional_total by using Eclipse or Rectangle or Freeform mask. The possibility of this masking is 20%. Ensure the masked area covers at least 10% and at most 50% of the foreground area. You can leverage the point_masks to determine the foreground area.
        
        # Apply masking to additional images with 50% probability
        if np.random.rand() < 0.5:
            # Select indices of additional images (after main images)
            additional_indices = list(range(10, img_per_seq))
            
            for idx in additional_indices:
                # Get current image shape
                img_height, img_width = images[idx].shape[:2]
                
                # Use point_mask to determine foreground area
                foreground_mask = point_masks[idx]
                foreground_area = np.sum(foreground_mask)
                total_area = img_height * img_width
                
                if foreground_area == 0:
                    continue  # Skip if no foreground
                
                # Try multiple times to generate a valid mask
                max_attempts = 10
                valid_mask_found = False
                
                for attempt in range(max_attempts):
                    # Randomly choose mask type: 0=rectangle, 1=ellipse, 2=free_form
                    mask_type = np.random.randint(0, 3)
                    
                    if mask_type == 2:  # Free form mask
                        # Use the smaller dimension for resolution to ensure mask fits
                        resolution = min(img_height, img_width)
                        generated_mask = free_form_mask(resolution)
                        # Resize to match image dimensions if needed
                        if resolution != img_height or resolution != img_width:
                            generated_mask = generated_mask.resize((img_width, img_height), Image.LANCZOS)
                    else:
                        # Use random_mask for rectangle/ellipse
                        generated_mask = random_mask((img_width, img_height), ratio=1, mask_full_image=False)
                    
                    # Convert mask to numpy array
                    mask_array = prepare_mask_and_masked_image(generated_mask)
                    mask_array = mask_array.squeeze().numpy()  # Remove batch dimension and convert to numpy
                    
                    # Calculate overlap between generated mask and foreground
                    masked_foreground_area = np.sum(mask_array * foreground_mask.astype(np.float32))
                    foreground_mask_ratio = masked_foreground_area / foreground_area
                    
                    # Check if the mask covers 10-50% of foreground area
                    if 0.1 <= foreground_mask_ratio <= 0.5:
                        valid_mask_found = True
                        break
                
                if valid_mask_found:
                    # Apply mask to all relevant data
                    mask_3d = mask_array[..., np.newaxis]  # Add channel dimension for broadcasting
                    
                    # Apply mask to images (set masked areas to 0)
                    masked_value = 0.0  # Black for images
                    images[idx] = images[idx] * (1 - mask_3d) + masked_value * mask_3d

                    # save images[idx] for visualization
                    # Image.fromarray((images[idx]).astype(np.uint8)).save(f"debug_masked_image_{idx}.png")
                    
                    # Apply mask to depths (set masked areas to 0)
                    if len(depths) > idx and depths[idx] is not None:
                        depths[idx] = depths[idx] * (1 - mask_array)
                    
                    # Apply mask to cam_points and world_points (set masked areas to 0)
                    if len(cam_points) > idx and cam_points[idx] is not None:
                        cam_points[idx] = cam_points[idx] * (1 - mask_3d)
                    
                    if len(world_points) > idx and world_points[idx] is not None:
                        world_points[idx] = world_points[idx] * (1 - mask_3d)
                    
                    # Update point_masks (masked areas become invalid)
                    if len(point_masks) > idx and point_masks[idx] is not None:
                        point_masks[idx] = point_masks[idx] * (1 - mask_array.astype(bool))
        
        # apply imgaug augmentation to additional images. Considering different kinds of blur, kinds of noise, jpeg compression, coarse dropout, each with a probability of 0.3 and a random scale set. 
        additional_indices = list(range(10, img_per_seq))
            
        for idx in additional_indices:
            
            # Apply augmentation to image
            # imgaug expects uint8 images
            img_uint8 = images[idx].astype(np.uint8)

            # if all pixels are black, skip augmentation and set extrinsics to identity
            if np.sum(point_masks[idx]) < 1e-5:
                extrinsics[idx] = np.eye(4, dtype=extrinsics[idx].dtype)
                continue

            # get foreground mask from image where pixels are not black
            foreground_mask = np.any(images[idx] > 5, axis=-1).astype(np.float32)
            
            # Apply the augmentation to images
            augmented_img = self._get_aug_pipeline()(image=img_uint8)

            # Blend augmented image with original using foreground mask to avoid altering black background too much
            foreground_mask_3d = foreground_mask[..., np.newaxis]
            augmented_img = images[idx] * (1 - foreground_mask_3d) + augmented_img * foreground_mask_3d

            images[idx] = augmented_img.astype(np.float32)

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "ids": np.array(list(range(total_images))),
            "camera_mask": mask_ids,
            "frame_num": len(extrinsics),
            "images": images,
            "base_colors": base_colors,
            "depths": depths if depths and depths[0] is not None else [],
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points if cam_points and cam_points[0] is not None else [],
            "world_points": world_points if world_points and world_points[0] is not None else [],
            "point_masks": point_masks if point_masks and point_masks[0] is not None else [],
            "original_sizes": original_sizes,
            "point_sonata": point_sonata,
            "initial_sonata_num": initial_sonata_num,
        }
        return batch
