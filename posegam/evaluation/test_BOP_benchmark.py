import os
import torch
import numpy as np
import json
import random
import logging
import warnings
import sys
import csv
import cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from posegam.models.posegam import PoseGAM
from posegam.utils.pose_enc import pose_encoding_to_extri_intri, extri_intri_to_pose_encoding
from posegam.utils.geometry import closed_form_inverse_se3
from posegam.training.data.dataset_util import *
import argparse
from scipy.spatial.transform import Rotation as R
from posegam.dependency import sonata
from pycocotools import mask as maskUtils
from posegam.evaluation.fix_BOP_translation import rendered_mask_image, load_mesh_with_textures
from tqdm import tqdm

# Suppress DINO v2 logs
logging.getLogger("dinov2").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="dinov2")

# Set computation precision
torch.set_float32_matmul_precision('highest')
torch.backends.cudnn.allow_tf32 = False


def _is_finite_array(arr):
    return np.all(np.isfinite(np.asarray(arr)))


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
    """Mirror of posegam.training.data.datasets.lmo.randomly_expand_binary_mask.
    Inlined here because that module has non-portable relative imports."""
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

def _analytical_pose_init(K_src, K_dst, R_src, T_src):
    """
    Compute an analytical initial estimate for (R_dst, T_dst) using
    the projection matrix approach with SVD polar decomposition.
    This is used as initialization for the PnP solver.
    """
    try:
        K_src = np.asarray(K_src, dtype=np.float64)
        K_dst = np.asarray(K_dst, dtype=np.float64)
        R_src = np.asarray(R_src, dtype=np.float64)
        T_src = np.asarray(T_src, dtype=np.float64).reshape(3,)
    except Exception:
        return np.asarray(R_src).copy(), np.asarray(T_src).copy()

    if not (_is_finite_array(K_src) and _is_finite_array(K_dst) and _is_finite_array(R_src) and _is_finite_array(T_src)):
        return R_src.copy(), T_src.copy()

    try:
        if abs(np.linalg.det(K_dst)) < 1e-12:
            return R_src.copy(), T_src.copy()
    except np.linalg.LinAlgError:
        return R_src.copy(), T_src.copy()

    P_src = K_src @ np.hstack((R_src, T_src.reshape(3, 1)))  # 3x4
    try:
        M = np.linalg.solve(K_dst, P_src)  # 3x4
    except np.linalg.LinAlgError:
        return R_src.copy(), T_src.copy()

    if not _is_finite_array(M):
        return R_src.copy(), T_src.copy()

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

    if not _is_finite_array(S):
        return R_src.copy(), T_src.copy()

    scale = np.mean(S[:2])
    if not np.isfinite(scale) or abs(scale) < 1e-12:
        return R_src.copy(), T_src.copy()

    T_init = m4 / (scale + 1e-8)
    if not (_is_finite_array(R_init) and _is_finite_array(T_init)):
        return R_src.copy(), T_src.copy()

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

    K_src = np.asarray(K_src, dtype=np.float64)
    K_dst = np.asarray(K_dst, dtype=np.float64)
    R_src = np.asarray(R_src, dtype=np.float64)
    T_src = np.asarray(T_src, dtype=np.float64).reshape(3,)

    if not (_is_finite_array(K_src) and _is_finite_array(K_dst) and _is_finite_array(R_src) and _is_finite_array(T_src)):
        print("Warning: Non-finite values in intrinsics/pose, returning original pose")
        return R_src, T_src

    try:
        if abs(np.linalg.det(K_dst)) < 1e-12:
            print("Warning: K_dst is singular, returning original pose")
            return R_src, T_src
    except np.linalg.LinAlgError:
        print("Warning: K_dst determinant failed, returning original pose")
        return R_src, T_src

    # ---- Step 1: Gather 3D test points at the correct scale ----
    if model_points is not None:
        # Use actual mesh vertices — subsample if too many for speed
        if hasattr(model_points, 'cpu'):
            model_points = model_points.cpu().numpy()
        model_points = model_points.astype(np.float64)
        pts_3d = model_points.copy()
    else:
        # Generate synthetic points scaled to match T_src magnitude
        # Object extent ~ ||T_src|| * tan(FOV/2); conservative: use ||T_src|| as scale
        scale = max(np.linalg.norm(T_src), 1.0)

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
    try:
        src_matrix_transformed = src_matrix @ np.linalg.inv(coord_transform)
    except np.linalg.LinAlgError:
        print("Warning: coordinate transform inversion failed, returning original pose")
        return R_src, T_src

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
    try:
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
    except cv2.error:
        success, rvec, tvec = False, None, None

    if not success:
        # Fallback: try SQPNP without initial guess
        try:
            success, rvec, tvec = cv2.solvePnP(
                pts_3d.reshape(-1, 1, 3),
                pts_2d.reshape(-1, 1, 2),
                K_dst.astype(np.float64),
                None,
                flags=cv2.SOLVEPNP_SQPNP
            )
        except cv2.error:
            success, rvec, tvec = False, None, None

    if not success:
        print("Warning: PnP failed completely, returning original pose")
        return R_src, T_src

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

    if rvec is None or tvec is None:
        print("Warning: PnP returned empty outputs, returning original pose")
        return R_src, T_src

    R_dst_transformed, _ = cv2.Rodrigues(rvec)
    T_dst_transformed = tvec.flatten()

    if not (_is_finite_array(R_dst_transformed) and _is_finite_array(T_dst_transformed)):
        print("Warning: Converted pose contains non-finite values, returning original pose")
        return R_src, T_src

    # ---- Step 6.5: Transform optimized pose back to original frame before returning ----
    dst_matrix_transformed = np.eye(4, dtype=np.float64)
    dst_matrix_transformed[:3, :3] = R_dst_transformed.astype(np.float64)
    dst_matrix_transformed[:3, 3] = T_dst_transformed.astype(np.float64)
    dst_matrix = dst_matrix_transformed @ coord_transform
    R_dst = dst_matrix[:3, :3]
    T_dst = dst_matrix[:3, 3]

    if not (_is_finite_array(R_dst) and _is_finite_array(T_dst)):
        print("Warning: Converted pose became non-finite after frame transform, returning original pose")
        return R_src, T_src

    # ---- Step 8: Compute and log reprojection error ----
    pts_reproj = (K_dst @ (R_dst @ pts_3d.T + T_dst.reshape(3, 1))).T
    pts_reproj_2d = pts_reproj[:, :2] / pts_reproj[:, 2:3]
    reproj_err = np.mean(np.linalg.norm(pts_2d - pts_reproj_2d, axis=1))
    max_err = np.max(np.linalg.norm(pts_2d - pts_reproj_2d, axis=1))
    print(f"  Intrinsic conversion: mean reproj err = {reproj_err:.4f} px, max = {max_err:.4f} px  ({len(pts_3d)} pts)")

    return R_dst, T_dst


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

def parse_camera_parameters(frame_data, image_width=512, image_height=512):
    """
    Parse camera parameters from BOP transforms.json format.
    Returns:
        intrinsics: 3x3 camera intrinsic matrix
        extrinsics: 4x4 extrinsic matrix for the view
    """
    # Parse intrinsics from frame
    fov_x = frame_data['camera_angle_x']  # Field of view in radians
    
    # Calculate focal length from FOV
    focal_length = image_width / (2.0 * np.tan(fov_x / 2.0))
    
    # Construct intrinsic matrix (assuming principal point at image center)
    intrinsics = np.array([
        [focal_length, 0, image_width / 2.0],
        [0, focal_length, image_height / 2.0],
        [0, 0, 1]
    ])
    
    # Transform matrix is already in the correct 4x4 format
    transform_matrix = np.array(frame_data['transform_matrix'])

    # Convert from Blender to OpenCV coordinate system
    blender_to_opencv = np.array([
        [1,  0,  0,  0],
        [0,  0,  1,  0],  # Y becomes Z
        [0, -1,  0,  0],  # Z becomes -Y
        [0,  0,  0,  1]
    ])
        
    return intrinsics, blender_to_opencv @ transform_matrix


def setup_args():
    """Set up command-line arguments for the BOP evaluation script."""
    parser = argparse.ArgumentParser(description='Test PoseGAM on BOP dataset')
    parser.add_argument('--min_num_images', type=int, default=24, help='Minimum number of images for a sequence')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames to use for testing')
    parser.add_argument('--output_name', type=str, help='Prefix for the output result CSV file', default='posegam')
    parser.add_argument('--BOP_dir', type=str, help='Path to BOP dataset directory', default='/ibex/tmp/TRELLIS-500K/BOP-data/')
    parser.add_argument('--BOP_dataset_name', type=str, help='Name of the BOP dataset', default='ycbv')
    parser.add_argument('--BOP_query_dir', type=str, help='Path to the BOP query directory (gigapose datasets/tmp)', default='/path/to/gigapose/gigaPose_datasets/datasets/tmp')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--model_path', type=str, help='Path to the PoseGAM model checkpoint', default='./posegam.pt')
    parser.add_argument('--whether_save', action='store_true', default=False, help='Whether to save the evaluation results')
    parser.add_argument('--rank_id', type=int, default=0, help='Rank ID for distributed processing (0-indexed)')
    parser.add_argument('--total_ranks', type=int, default=1, help='Total number of ranks for distributed processing')
    return parser.parse_args()


def load_model(device, model_path):
    """
    Load the PoseGAM model.

    Args:
        device: Device to load the model on
        model_path: Path to the model checkpoint

    Returns:
        Loaded PoseGAM model
    """
    print("Initializing and loading PoseGAM model...")
    # "depths", "cam_points", "world_points", "point_masks", "base_colors"
    model = PoseGAM(enable_mask=True, input_keys=["world_points", "point_masks"])
    print(f"USING {model_path}")
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    model = model.to(device)
    return model


def set_random_seeds(seed):
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_cnos_masks(mask_path, dataset_name):
    """
    Load the CNOS-FastSAM detection masks used as the query-image segmentation.

    Args:
        mask_path: Path to the directory containing CNOS mask JSON files
        dataset_name: Name of the dataset (e.g., 'ycbv', 'lmo')

    Returns:
        Dictionary mapping (scene_id, image_id, category_id) to list of mask entries with scores
    """
    dict_masks = {'icbin':'cnos-fastsam_icbin-test_f21a9faf-7ef2-4325-885f-f4b6460f4432.json',
                  'lmo': 'cnos-fastsam_lmo-test_3cb298ea-e2eb-4713-ae9e-5a7134c5da0f.json',
                  'tless': 'cnos-fastsam_tless-test_8ca61cb0-4472-4f11-bce7-1362a12d396f.json',
                  'tudl': 'cnos-fastsam_tudl-test_c48a2a95-1b41-4a51-9920-a667cb3d7149.json',
                  'ycbv': 'cnos-fastsam_ycbv-test_f4f2127c-6f59-447c-95b3-28e1e591f1a1.json'
                  }
    
    mask_file = os.path.join(mask_path, 'cnos-fastsam', dict_masks[dataset_name])
    
    # Load the CNOS mask JSON file
    with open(mask_file, 'r') as f:
        cnos_data = json.load(f)
    
    # Organize masks by (scene_id, image_id, category_id)
    organized_masks = {}
    for entry in cnos_data:
        key = (entry['scene_id'], entry['image_id'], entry['category_id'])
        if key not in organized_masks:
            organized_masks[key] = []
        organized_masks[key].append(entry)
    
    # Sort masks by score (descending) for each key
    for key in organized_masks:
        organized_masks[key].sort(key=lambda x: x['score'], reverse=True)
    
    return organized_masks
    

def load_BOP_sequences(BOP_dir, min_num_images, test_json_path):
    """
    Load BOP sequences from the dataset directory.

    Args:
        BOP_dir: Path to BOP renders3 directory
        min_num_images: Minimum number of images per sequence

    Returns:
        Dictionary mapping sequence names to their frame data
    """
    sequences = {}
    
    # Get all sequence directories
    all_sequences = []
    for d in os.listdir(BOP_dir):
        if "obj_" in d:
            continue
        for dd in os.listdir(os.path.join(BOP_dir, d)):
            all_sequences.append(os.path.join(d, dd))
    
    # Filter sequences that have both transforms.json and mesh.npz
    valid_sequences = []
    for seq_name in all_sequences:
        seq_dir = os.path.join(BOP_dir, seq_name)
        transforms_file = os.path.join(seq_dir, "transforms.json")
        
        if os.path.exists(transforms_file):
            valid_sequences.append(seq_name)
    
    eval_sequences = valid_sequences
    
    print(f"Total sequences found: {len(all_sequences)}")
    print(f"Valid sequences (with both transforms.json and mesh.npz): {len(valid_sequences)}")
    
    for seq_name in eval_sequences:
        seq_dir = os.path.join(BOP_dir, seq_name)
        transforms_file = os.path.join(seq_dir, "transforms.json")
        
        try:
            with open(transforms_file, 'r') as f:
                transforms_data = json.load(f)
            
            frames = transforms_data.get('frames', [])
            
            if len(frames) >= min_num_images:
                sequences[seq_name] = frames
                
        except Exception as e:
            print(f"Error loading sequence {seq_name}: {e}")
            continue

    # filter out sequences that is not in test_json_path
    # Load test targets from JSON file
    with open(test_json_path, 'r') as f:
        test_targets = json.load(f)
    
    # Create a dict of valid (scene_id, im_id, obj_id) tuples mapping to inst_count for fast lookup
    valid_targets = {}
    for target in test_targets:
        key = (target['scene_id'], target['im_id'], target['obj_id'])
        assert key not in valid_targets, f"Duplicate key found in test_targets: {key}"
        valid_targets[key] = target['inst_count']
    
    # Filter sequences
    filtered_sequences = {}
    for seq_name in list(sequences.keys()):
        scene_id = int(seq_name.split('/')[0].split('_')[0])
        img_id = int(seq_name.split('/')[0].split('_')[1])
        obj_id = int(seq_name.split('/')[1])
        
        # Check if this sequence is in the test targets
        key = (scene_id, img_id, obj_id)
        if key in valid_targets:
            seq_data = sequences[seq_name].copy()
            seq_data.append(valid_targets[key])
            filtered_sequences[seq_name] = seq_data
    
    print(f"Filtered sequences: {len(sequences)} -> {len(filtered_sequences)} (based on test_targets_bop19.json)")
    
    return filtered_sequences


def process_sequence(model, seq_name, seq_frames, BOP_dir, min_num_images, num_frames, device, dtype, whether_save, known_camera_save_dir=None, unknown_camera_save_dir=None, dataset_name=None, query_dir=None, inst_id=None, cnos_masks=None):
    """
    Process a single sequence and compute pose errors.

    Args:
        model: PoseGAM model
        seq_name: Sequence name
        seq_frames: List of frame data from transforms.json
        BOP_dir: BOP dataset directory
        min_num_images: Minimum number of images required
        num_frames: Number of frames to sample
        device: Device to run on
        dtype: Data type for model inference
        whether_save: Whether to save evaluation results
        known_camera_save_dir: Directory to save known camera results (optional)
        unknown_camera_save_dir: Directory to save unknown camera results (optional)

    Returns:
        dict: Dictionary containing scene_id, im_id, obj_id, score, R, t, time or None if processing failed
    """
    if len(seq_frames) < min_num_images:
        return None
    

    num_main = num_frames

    # Sample IDs for main folder
    # main_ids = np.random.choice(len(seq_frames), num_main, replace=False)
    if num_main > 0:
        # Extract camera positions from transform matrices for FPS
        camera_positions = []
        for frame_data in seq_frames:
            transform_matrix = np.array(frame_data['transform_matrix'])
            # Extract camera position (translation part of transform matrix)
            camera_position = transform_matrix[:3, 3]
            camera_positions.append(camera_position)
        
        camera_positions = np.array(camera_positions)
        
        # Use FPS to sample diverse camera viewpoints
        main_ids = farthest_point_sampling(camera_positions, num_main)
    else:
        main_ids = np.array([], dtype=int)
    print("Main folder image ids", main_ids)

    # Collect frames from main folder and additional folders
    all_frames = []
    all_sources = []  # Track which folder each frame comes from
    all_data_roots = []  # Track data root for each frame
    all_depth_roots = []  # Track depth root for each frame
    
    # Add frames from main folder
    main_frames = [seq_frames[i] for i in main_ids]
    all_frames.extend(main_frames)
    all_sources.extend([dataset_name] * len(main_frames))
    all_data_roots.extend([BOP_dir] * len(main_frames))
    all_depth_roots.extend([BOP_dir] * len(main_frames))
    
    # Load images and ground truth poses
    images, base_colors, normals, depths, cam_points, world_points, point_masks = [], [], [], [], [], [], []
    gt_extri = []
    gt_intri = []
    
    for i, (frame_data, source_folder, data_root, depth_root) in enumerate(zip(all_frames, all_sources, all_data_roots, all_depth_roots)):
        # Get image path from appropriate data root
        image_path = os.path.join(data_root, seq_name, frame_data['file_path'])

        base_color_path = os.path.join(data_root.replace(source_folder, source_folder+'-color'), seq_name, frame_data['file_path'])
        assert f'{source_folder}-color' in base_color_path, f"Expected '{source_folder}-color' in base color path, but got: {base_color_path}"

        normal_path = os.path.join(data_root.replace(source_folder, source_folder+'-normal'), seq_name, frame_data['file_path'])
        assert f'{source_folder}-normal' in normal_path, f"Expected '{source_folder}-normal' in normal path, but got: {normal_path}"

        # Get depth path from appropriate depth root
        # randomly choose seq_name under sample folder
        depth_path = os.path.join(depth_root, seq_name, frame_data['file_path']).replace(".png", "_depth.png")

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return None
            
        # Parse camera parameters
        from PIL import Image
        with Image.open(image_path) as img:
            image_width, image_height = img.size
            
        intrinsics, extrinsics = parse_camera_parameters(
            frame_data, image_width=image_width, image_height=image_height
        )
        
        # Load and preprocess images
        image = read_image_cv2(image_path)
        depth_map = read_depth(depth_path, 1.0)
        base_color = read_image_cv2(base_color_path)
        normal = read_image_cv2(normal_path)

        # Set the value of pure white pixels to 0 which are the background
        depth_map[depth_map == 65535] = 0

        # normalize depth map to [0, 1] range
        depth_map = depth_map / 65535.0
        
        # Convert normalized depth to true depth values using annotation depth range
        depth_info = frame_data["depth"]
        depth_min = depth_info["min"]
        depth_max = depth_info["max"]
        # Only apply depth scaling to valid (non-zero) pixels
        valid_mask = depth_map != 0
        depth_map[valid_mask] = depth_map[valid_mask] * (depth_max - depth_min) + depth_min

        # Crop images around center to make width = height
        current_h, current_w = image.shape[:2]
        if current_w != current_h:
            crop_size = min(current_w, current_h)
            
            # Calculate crop offsets to center the crop
            start_y = (current_h - crop_size) // 2
            start_x = (current_w - crop_size) // 2
            
            # Crop all images to square
            image = image[start_y:start_y + crop_size, start_x:start_x + crop_size]
            depth_map = depth_map[start_y:start_y + crop_size, start_x:start_x + crop_size]
            base_color = base_color[start_y:start_y + crop_size, start_x:start_x + crop_size]
            normal = normal[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            # Update intrinsics: adjust principal point
            intrinsics[0, 2] -= start_x  # cx
            intrinsics[1, 2] -= start_y  # cy
            
            # Update image dimensions
            image_width = crop_size
            image_height = crop_size

        aspect_ratio = image_height / image_width
        short_size = int(518 * aspect_ratio)
        small_size = 14
        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size
        target_size = np.array([short_size, 518])

        image, base_color, normal, depth_map, intrinsics, _ = resize_image_depth_and_intrinsic(
            image, base_color, normal, depth_map, intrinsics, target_size, np.array([image_height, image_width]), track=None,
            rescale_aug=False
        )

        # Ensure final crop to target shape
        image, base_color, normal, depth_map, intrinsics, _ = crop_image_depth_and_intrinsic_by_pp(
            image, base_color, normal, depth_map, intrinsics, target_size, track=None, filepath=None, strict=True,
        )

        normal = (normal / 255.0 - 0.5) * 2.0
        norm = np.linalg.norm(normal, axis=-1, keepdims=True)
        normal = normal / (norm + 1e-8)

        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extrinsics, intrinsics)
        )

        images.append(torch.from_numpy(image).float().to(device).unsqueeze(0).div(255))  # [1, 3, H, W]
        base_colors.append(torch.from_numpy(base_color if dataset_name not in ["tless"] else image).float().to(device).unsqueeze(0).div(255))
        normals.append(torch.from_numpy(normal).float().to(device).unsqueeze(0))
        depths.append(torch.from_numpy(depth_map).float().to(device).unsqueeze(-1).unsqueeze(0))
        cam_points.append(torch.from_numpy(cam_coords_points).float().to(device).unsqueeze(0))
        world_points.append(torch.from_numpy(world_coords_points).float().to(device).unsqueeze(0))
        point_masks.append(torch.from_numpy(point_mask).float().to(device).unsqueeze(-1).unsqueeze(0))

        gt_extri.append(torch.from_numpy(extrinsics).float().to(device))
        gt_intri.append(torch.from_numpy(intrinsics).float().to(device))

    # add query frames (crop and padding to ensure the same principal point in intrinsics)

    query_image_path = os.path.join(query_dir, dataset_name+'_image_wise', 'test' + ('_primesense' if dataset_name == "tless" else ''), seq_name.split('/')[0] + ".rgb.png")
    query_image = read_image_cv2(query_image_path)
    query_image_size = query_image.shape[:2]

    obj_name = seq_name.split('/')[1]
    scene_id = int(seq_name.split('/')[0].split('_')[0])
    img_id = int(seq_name.split('/')[0].split('_')[1])
    obj_id = int(obj_name)
    
    # Get the object mask for this instance from the CNOS-FastSAM detections
    key = (scene_id, img_id, obj_id)

    if key not in cnos_masks:
        return None
    mask_entries = cnos_masks[key]  # Already sorted by score (descending)

    # Select the inst_id-th highest scoring mask
    if inst_id >= len(mask_entries):
        return None
    selected_mask_entry = mask_entries[inst_id]

    # Decode the mask from CNOS format
    segmentation = selected_mask_entry['segmentation']
    binary_mask = maskUtils.decode(maskUtils.frPyObjects(
        segmentation, segmentation["size"][0], segmentation["size"][1]
    ))

    binary_mask = randomly_expand_binary_mask(
        binary_mask,
        expand_prob=1.0,
        target_area_ratio_range=(4.19, 4.19),
        extreme_expand_prob=0.0,
    ).astype(binary_mask.dtype)
    query_image = query_image * binary_mask[..., None]

    camera_info = json.load(open(os.path.join(query_dir, dataset_name+'_image_wise', 'test' + ('_primesense' if dataset_name == "tless" else ''), seq_name.split('/')[0] + ".camera.json"), 'r'))["cam_K"]
    camera_principal_point = (camera_info[2], camera_info[5])
    

    # the camera_principal_point is not exactly at the center of the image, we need to shift and pad the query image to offset this
    # Calculate center of the query image
    query_h, query_w = query_image_size
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
    
    # Update the intrinsic matrix to reflect the centered principal point
    query_intrinsics = np.array([
        [camera_info[0], 0, query_w / 2.0],
        [0, camera_info[4], query_h / 2.0],
        [0, 0, 1]
    ])

    # Crop the query_image_centered to the foreground bounding box
    centered_h, centered_w = query_image_centered.shape[:2]
    
    # Find bounding box of the binary mask
    mask_rows = np.any(binary_mask_centered, axis=1)
    mask_cols = np.any(binary_mask_centered, axis=0)
    
    if not mask_rows.any() or not mask_cols.any():
        # If mask is empty, fall back to using the full image
        print(f"Warning: Empty mask for {seq_name}, using full image")
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
    
    # Update intrinsics: adjust principal point based on crop offset from the centered image
    query_intrinsics = np.array([
        [camera_info[0], 0, query_intrinsics[0, 2] - crop_x_min],
        [0, camera_info[4], query_intrinsics[1, 2] - crop_y_min],
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
    
    # Update intrinsics for the center crop
    query_intrinsics_final = query_intrinsics_resized.copy()
    query_intrinsics_final[0, 2] -= start_x  # Adjust cx
    query_intrinsics_final[1, 2] -= start_y  # Adjust cy
    
    # Convert query image to tensor and append to images list
    query_image_tensor = torch.from_numpy(query_image_final).float().to(device).unsqueeze(0).div(255)

    images.append(query_image_tensor)
    # Append dummy data for other modalities (query doesn't have these)
    base_colors.append(query_image_tensor)
    normals.append(torch.zeros_like(normals[0]))
    depths.append(torch.zeros_like(depths[0]))
    cam_points.append(torch.zeros_like(cam_points[0]))
    world_points.append(torch.zeros_like(world_points[0]))
    point_masks.append(torch.zeros_like(point_masks[0]))
    
    # Append ground truth camera parameters (dummy extrinsic, actual intrinsic for query)
    gt_extri.append(gt_extri[0])  # Dummy extrinsic - this is what we want to predict
    gt_intri.append(torch.from_numpy(query_intrinsics_final).float().to(device))  # Actual query intrinsic

    gt_extri = torch.stack(gt_extri, dim=0)
    gt_intri = torch.stack(gt_intri, dim=0)

    images = torch.stack(images, dim=1).permute(0, 1, 4, 2, 3)  # [1, S, 3, H, W]
    base_colors = torch.stack(base_colors, dim=1).permute(0, 1, 4, 2, 3)  # [1, S, 3, H, W]
    normals = torch.stack(normals, dim=1).permute(0, 1, 4, 2, 3)  # [1, S, 3, H, W]
    depths = torch.stack(depths, dim=1).permute(0, 1, 4, 2, 3)  # [1, S, 1, H, W]
    cam_points = torch.stack(cam_points, dim=1).permute(0, 1, 4, 2, 3)  # [1, S, 3, H, W]
    world_points = torch.stack(world_points, dim=1).permute(0, 1, 4, 2, 3)  # [1, S, 3, H, W]
    point_masks = torch.stack(point_masks, dim=1).permute(0, 1, 4, 2, 3)  # [1, S, 1, H, W]

    # Prepare model inputs aligned with trainer.py format
    B = 1  # batch size
    S = len(all_frames) + 1  # actual sequence length
    
    # Convert numpy arrays to tensors and add batch dimension
    gt_extri_tensor = gt_extri.unsqueeze(0)  # [1, S, 4, 4]
    gt_intri_tensor = gt_intri.unsqueeze(0)  # [1, S, 3, 3]
    
    # Create camera mask aligned with posegam_dataset.py logic
    camera_params = extri_intri_to_pose_encoding(
        gt_extri_tensor, gt_intri_tensor, images.shape[-2:]
    )
    
    # Apply masking logic like posegam_dataset.py
    total_images = len(all_frames) + 1
    mask_ids = np.zeros(total_images, dtype=np.int32)
    
    # For the main folder images, apply random masking
    main_images_count = num_main

    mask_ids[:num_main] = 1  # First 10 images from main folder are known

    camera_mask = torch.from_numpy(mask_ids).float().to(device).unsqueeze(0)  # [1, S]

    point_sonata = dict()
    # point_sonata['coord'] = surface[:, :3]
    # point_sonata['normal'] = surface[:, 3:6]
    # point_sonata['color'] = surface[:, 6:9]

    # point_sonata = sonata.transform.default()(point_sonata)

    valid_coords = []
    valid_normals = []
    valid_colors = []

    for i in range(num_main):
        # Get arrays for this image
        current_world_pts = world_points[0][i].permute(1, 2, 0)  # Shape: (H, W, 3)
        current_normals = normals[0][i].permute(1, 2, 0)         # Shape: (H, W, 3)
        current_colors = base_colors[0][i].permute(1, 2, 0)      # Shape: (H, W, 3)
        current_mask = point_masks[0][i].permute(1, 2, 0).bool().squeeze()        # Shape: (H, W)

        # Apply downsampling if specified
        current_world_pts, current_normals, current_colors, current_mask = downsample_point_arrays(
            current_world_pts, current_normals, current_colors, current_mask, 
            7
        )
        
        # Extract valid points using the mask
        valid_world_pts = current_world_pts[current_mask]  # Shape: (N_valid, 3)
        valid_normal_pts = current_normals[current_mask]   # Shape: (N_valid, 3)
        valid_color_pts = current_colors[current_mask]     # Shape: (N_valid, 3)
        
        valid_coords.append(valid_world_pts.cpu().numpy())
        valid_normals.append(valid_normal_pts.cpu().numpy())
        valid_colors.append(valid_color_pts.cpu().numpy())

    point_sonata['coord'] = np.concatenate(valid_coords, axis=0)
    point_sonata['normal'] = np.concatenate(valid_normals, axis=0)
    point_sonata['color'] = np.concatenate(valid_colors, axis=0)

    initial_sonata_num = torch.from_numpy(np.array(point_sonata['coord'].shape[0]).astype(np.int32)).unsqueeze(0).to(device)

    point_sonata = sonata.transform.default()(point_sonata)

    for key in point_sonata.keys():
        if isinstance(point_sonata[key], torch.Tensor):
            point_sonata[key] = point_sonata[key].cuda(non_blocking=True)
    
    # Add batch dimension to images if not already present
    if len(images.shape) == 4:
        images = images.unsqueeze(0)  # [1, S, 3, H, W]
        base_colors = base_colors.unsqueeze(0)  # [1, S, 3, H, W]

    # Run PoseGAM inference
    try:
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images, camera_params, camera_mask, world_points, point_masks, base_colors, point_sonata, initial_sonata_num)

                mask_recon_prob = torch.sigmoid(predictions["mask_recon"][0, -1])

                # Compute inverse-transformed predicted mask aligned with binary_mask_centered
                # Reverses: (1) center-crop to target_size, (2) uniform resize, (3) bbox crop
                target_h_val, target_w_val = target_size[0], target_size[1]

                # mask_recon_prob: (1, target_h, target_w)
                # Step 1: undo center-crop — pad back to (new_h, new_w)
                mask_recon_padded = torch.zeros(
                    1, new_h, new_w,
                    device=device, dtype=mask_recon_prob.dtype
                )
                mask_recon_padded[
                    :,
                    start_y:start_y + target_h_val,
                    start_x:start_x + target_w_val
                ] = mask_recon_prob

                # Step 2: undo resize — interpolate back to (crop_h, crop_w)
                mask_recon_unresized = torch.nn.functional.interpolate(
                    mask_recon_padded.unsqueeze(0),  # (1, 1, new_h, new_w)
                    size=(crop_h, crop_w),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # (1, crop_h, crop_w)

                # Step 3: undo bbox crop — place into (query_h, query_w) canvas
                mask_recon_canvas = torch.zeros(
                    1, query_h, query_w,
                    device=device, dtype=mask_recon_prob.dtype
                )
                mask_recon_canvas[
                    :,
                    crop_y_min:crop_y_min + crop_h,
                    crop_x_min:crop_x_min + crop_w
                ] = mask_recon_unresized

                # Reshape to (query_h, query_w, 1) — same layout as binary_mask_centered
                mask_recon_aligned = mask_recon_canvas.permute(1, 2, 0)  # (query_h, query_w, 1)

        with torch.cuda.amp.autocast(dtype=torch.float64):
            extrinsic, normal_intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            normal_pred_extrinsic = extrinsic[0]
    except Exception as e:
        print(f"PoseGAM inference failed: {e}")
        return None

    # Parse sequence information for CSV output
    scene_id = seq_name.split('/')[0].split('_')[0]
    img_id = seq_name.split('/')[0].split('_')[1]
    obj_id = seq_name.split('/')[1]
    score = 1
    time = -1

    model_info = json.load(open(os.path.join(query_dir.replace('tmp', dataset_name), 'models' + ('_cad' if dataset_name in ["tless"] else ''), 'models_info.json'), 'r'))[f'{obj_id}']
    model_initial_size = [model_info['size_x'], model_info['size_y'], model_info['size_z']]
    model_scale = max(model_initial_size)

    predictions_to_test = [(normal_intrinsic, normal_pred_extrinsic, 0)]  # (intrinsic, extrinsic, rotation_angle)
    
    best_result = None
    best_final_iou = -1.0

    vert_pre, face_pre = load_mesh_with_textures(os.path.join(BOP_dir, 'obj_'+f'{int(obj_id):06d}', 'mesh.glb'))
    
    for intrinsic, pred_extrinsic, rotation_angle in predictions_to_test:
        print(f"\n{'='*60}")
        print(f"Processing {rotation_angle}° rotation prediction")
        print(f"{'='*60}")
        
        # Convert camera-to-world to world-to-camera
        R_c2w = pred_extrinsic[-1][:3, :3].cpu().numpy()
        t_c2w = pred_extrinsic[-1][:3, 3].cpu().numpy()

        convert_matrix_1 = np.array([
            [1,  0,  0,  0],
            [0,  -1,  0,  0],
            [0,  0,  -1,  0],
            [0,  0,  0,  1]
        ])

        convert_matrix_2 = np.array([
            [1,  0,  0,  0],
            [0,  0,  -1,  0],
            [0,  1,  0,  0],
            [0,  0,  0,  1]
        ])

        # combine R and t into a 4x4 matrix
        t_c2w_homogeneous = np.eye(4)
        t_c2w_homogeneous[:3, :3] = R_c2w
        t_c2w_homogeneous[:3, 3] = t_c2w

        trans = convert_matrix_1 @ np.linalg.inv(t_c2w_homogeneous) @ np.linalg.inv(convert_matrix_2)

        R = trans[:3, :3]
        t = trans[:3, 3]

        cur_intrin = intrinsic[0, -1].cpu().numpy()
        
        tar_intrin = query_intrinsics_final

        R, t = convert_pose_between_intrinsics(
            cur_intrin,
            tar_intrin,
            R,
            t,
            model_points=vert_pre,
        )

        R = torch.from_numpy(R).float().to(device)
        t = torch.from_numpy(t).float().to(device)

        t.requires_grad = True

        optimizer = torch.optim.Adam([t], lr=1e-1)

        # Optimization parameters
        max_iters = 200
        check_interval = 30  # Check IoU improvement every N iterations
        min_iou_improvement = 1e-4  # Minimum IoU improvement to continue

        iter = 0
        best_iou = 0.0
        last_check_iou = 0.0
        best_R = R.clone()
        best_t = t.clone().detach()
        opt_desc = f"translation ({rotation_angle}° rotation)"
        pbar = tqdm(total=max_iters, desc=f"Optimizing {opt_desc}")
        
        while iter < max_iters:
            optimizer.zero_grad()


            verify_matrix = torch.eye(4).float().to(t.device)
            verify_matrix[:3, :3] = R
            verify_matrix[:3, 3] = t

            ren_height, ren_width = query_h, query_w
            ren_fx = camera_info[0]
            ren_fy = camera_info[4]
            render_mask_img = rendered_mask_image(None, None, verify_matrix, model_scale, width=ren_width, height=ren_height, fx=ren_fx, fy=ren_fy, vert_pre=vert_pre, face_pre=face_pre)
            
            # if the rendered mask is nearly all zeros or all ones, manually set a t that can ensure the object can be projected within the image frame and then continue optimization
            if iter == 0:
                total_pixels = render_mask_img.numel()
                white_ratio = (render_mask_img > 0.5).sum().item() / total_pixels
                black_ratio = (render_mask_img < 0.5).sum().item() / total_pixels

                # Check if object is clamped by image border (mask=1 pixels on border)
                h, w, _ = render_mask_img.shape
                border_mask = torch.zeros_like(render_mask_img, dtype=torch.bool)
                border_mask[0, :] = True  # Top border
                border_mask[-1, :] = True  # Bottom border
                border_mask[:, 0] = True  # Left border
                border_mask[:, -1] = True  # Right border
                
                border_pixels_activated = ((render_mask_img > 0.5) & border_mask).sum().item()
                is_clamped_by_border = border_pixels_activated > 0

                # Check if mask is nearly all white (>99%) or all black (>99%)
                if white_ratio > 0.99 or black_ratio > 0.99 or is_clamped_by_border:
                    if is_clamped_by_border:
                        print(f"Warning: Object is clamped by image border ({border_pixels_activated} border pixels activated). Adjusting translation...")
                    else:
                        print(f"Warning: Rendered mask is nearly {'white' if white_ratio > 0.99 else 'black'} ({white_ratio:.2%} white). Adjusting translation...")

                    # Get object center in world coordinates from ground truth / predicted mask
                    if mask_recon_aligned is not None:
                        gt_mask_img = mask_recon_aligned.to(device)
                    else:
                        gt_mask_img = torch.from_numpy(binary_mask_centered.astype(np.float32)).to(device)
                    
                    # Find the center of the ground truth mask in image coordinates
                    mask_coords = torch.nonzero(gt_mask_img > 0.5, as_tuple=False).float()
                    if len(mask_coords) > 0:
                        mask_center_y = mask_coords[:, 0].mean()
                        mask_center_x = mask_coords[:, 1].mean()
                        
                        # Convert image coordinates to normalized device coordinates
                        h, w = gt_mask_img.shape[:2]
                        ndc_x = (mask_center_x - tar_intrin[0, 2]) / tar_intrin[0, 0]
                        ndc_y = (mask_center_y - tar_intrin[1, 2]) / tar_intrin[1, 1]
                        
                        # Estimate appropriate depth based on object size and focal length
                        # Use the diagonal of the mask bounding box as a proxy for object size
                        mask_y_min = mask_coords[:, 0].min()
                        mask_y_max = mask_coords[:, 0].max()
                        mask_x_min = mask_coords[:, 1].min()
                        mask_x_max = mask_coords[:, 1].max()
                        
                        bbox_height = (mask_y_max - mask_y_min).item()
                        bbox_width = (mask_x_max - mask_x_min).item()
                        bbox_diagonal = np.sqrt(bbox_height**2 + bbox_width**2)
                        
                        # Estimate depth: larger objects in image should be closer
                        # Use a heuristic: depth inversely proportional to bbox size
                        focal_length = tar_intrin[0, 0].item()
                        estimated_depth = (model_scale * focal_length) / (bbox_diagonal + 1e-6)
                        
                        # Clamp depth to reasonable range (0.5 to 5.0 times model scale)
                        estimated_depth = np.clip(estimated_depth, 0.5 * model_scale, 5.0 * model_scale)
                        
                        # Compute 3D translation in camera coordinates
                        t_x = ndc_x.item() * estimated_depth
                        t_y = ndc_y.item() * estimated_depth
                        t_z = estimated_depth
                        
                        # Update translation with the manually computed values
                        with torch.no_grad():
                            t[0] = t_x / model_scale * 2.1
                            t[1] = t_y / model_scale * 2.1
                            t[2] = t_z / model_scale * 2.1
                        
                        print(f"Manually set translation to: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")
                        
                        # Re-render with adjusted translation
                        verify_matrix = torch.eye(4).float().to(t.device)
                        verify_matrix[:3, :3] = R
                        verify_matrix[:3, 3] = t
                        ren_height, ren_width = query_h, query_w
                        ren_fx = camera_info[0]
                        ren_fy = camera_info[4]
                        render_mask_img = rendered_mask_image(None, None, verify_matrix, model_scale, width=ren_width, height=ren_height, fx=ren_fx, fy=ren_fy, vert_pre=vert_pre, face_pre=face_pre)
                        
                        # Verify the adjustment worked
                        new_white_ratio = (render_mask_img > 0.5).sum().item() / total_pixels
                        print(f"After adjustment: {new_white_ratio:.2%} white pixels")

            if mask_recon_aligned is not None:
                gt_mask_img = mask_recon_aligned.to(device)
            else:
                gt_mask_img = torch.from_numpy(binary_mask_centered.astype(np.float32)).to(device)

            # Differentiable IoU loss using soft masks
            intersection = (render_mask_img * gt_mask_img).sum()
            union = render_mask_img.sum() + gt_mask_img.sum() - intersection
            iou_loss = 1.0 - (intersection + 1e-6) / (union + 1e-6)
            current_iou = 1.0 - iou_loss.item()
            
            # Track best IoU and save best parameters
            if current_iou > best_iou:
                best_iou = current_iou
                best_R = R.detach().clone()
                best_t = t.clone().detach()
            
            # Differentiable 2D Chamfer distance using random sampling from foreground
            chamfer_loss = torch.tensor(0.0, device=device)
            if render_mask_img.sum() > 0.1 and gt_mask_img.sum() > 0.1:
                # Create coordinate grids
                h, w, _ = render_mask_img.shape
                y_coords = torch.arange(h, dtype=torch.float32, device=device).view(-1, 1).expand(h, w)
                x_coords = torch.arange(w, dtype=torch.float32, device=device).view(1, -1).expand(h, w)
                
                # Flatten and create coordinate tensors
                coords = torch.stack([y_coords.flatten(), x_coords.flatten()], dim=1)  # [H*W, 2]
                
                # Use soft masks as sampling weights
                render_weights = render_mask_img.flatten()  # [H*W]
                gt_weights = gt_mask_img.flatten()  # [H*W]

                # Random sampling from foreground area based on mask probabilities
                max_points = 500
                
                if render_weights.sum() > 0:
                    # Normalize to get probability distribution
                    render_probs = render_weights / (render_weights.sum() + 1e-8)
                    
                    # Random sampling based on probabilities (multinomial)
                    num_samples = min(max_points, (render_weights > 0.).sum().item())
                    if num_samples > 0:
                        sampled_indices = torch.multinomial(
                            render_probs, 
                            num_samples=num_samples, 
                            replacement=False
                        )
                        render_coords = coords[sampled_indices]
                        render_sample_weights = render_weights[sampled_indices]
                    else:
                        render_coords = None
                        render_sample_weights = None
                else:
                    render_coords = None
                    render_sample_weights = None
                
                if gt_weights.sum() > 0:
                    # Normalize to get probability distribution
                    gt_probs = gt_weights / (gt_weights.sum() + 1e-8)
                    
                    # Random sampling based on probabilities (multinomial)
                    num_samples = min(max_points, (gt_weights > 0.).sum().item())
                    if num_samples > 0:
                        sampled_indices = torch.multinomial(
                            gt_probs, 
                            num_samples=num_samples, 
                            replacement=False
                        )
                        gt_coords = coords[sampled_indices]
                        gt_sample_weights = gt_weights[sampled_indices]
                    else:
                        gt_coords = None
                        gt_sample_weights = None
                else:
                    gt_coords = None
                    gt_sample_weights = None
                
                # Compute soft Chamfer distance
                if render_coords is not None and gt_coords is not None and len(render_coords) > 0 and len(gt_coords) > 0:
                    # Render -> GT: weighted distance from rendered mask to GT mask
                    dist_render_to_gt = torch.cdist(render_coords, gt_coords, p=2)  # [N_render, N_gt]
                    
                    # Use soft minimum with temperature for differentiability
                    temp_chamfer = 0.1
                    weights_render_to_gt = torch.softmax(-dist_render_to_gt / temp_chamfer, dim=1)  # [N_render, N_gt]
                    soft_min_dist_render_to_gt = (weights_render_to_gt * dist_render_to_gt).sum(dim=1)  # [N_render]
                    chamfer_render_to_gt = (soft_min_dist_render_to_gt * render_sample_weights).sum() / (render_sample_weights.sum() + 1e-8)
                    
                    # GT -> Render: weighted distance from GT mask to rendered mask
                    dist_gt_to_render = dist_render_to_gt.t()  # [N_gt, N_render]
                    weights_gt_to_render = torch.softmax(-dist_gt_to_render / temp_chamfer, dim=1)  # [N_gt, N_render]
                    soft_min_dist_gt_to_render = (weights_gt_to_render * dist_gt_to_render).sum(dim=1)  # [N_gt]
                    chamfer_gt_to_render = (soft_min_dist_gt_to_render * gt_sample_weights).sum() / (gt_sample_weights.sum() + 1e-8)
                    
                    # Symmetric Chamfer distance with emphasis on render->GT (handles occlusion better)
                    chamfer_loss = (0.6 * chamfer_render_to_gt + 0.4 * chamfer_gt_to_render) / 100
            
            # Combine losses with weights
            loss = 0.6 * iou_loss + 0.4 * chamfer_loss  # Scale chamfer to similar magnitude

            loss.backward()
            optimizer.step()
            
            # Update progress bar with loss value
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}', 
                'iou': f'{current_iou:.4f}',
                'best_iou': f'{best_iou:.4f}',
                'chamfer': f'{chamfer_loss.item():.2f}'
            })
            pbar.update(1)
            
            iter += 1
            
            # Check for IoU improvement every check_interval iterations
            if iter % check_interval == 0 and iter > 0:
                iou_improvement = current_iou - last_check_iou
                
                # Break if IoU has not improved enough
                if iou_improvement < min_iou_improvement:
                    print(f"\nEarly stopping at iteration {iter}: IoU improvement ({iou_improvement:.6f}) < threshold ({min_iou_improvement})")
                    break
                
                last_check_iou = current_iou
        
        pbar.close()

        # Restore best R and T
        R = best_R
        t = best_t
        print(f"\nUsing best parameters with IoU: {best_iou:.4f}")

        R = R.detach().cpu().numpy()
        t = t.detach().cpu().numpy()

        t = t * model_scale / 2.1

        # Format R and t as row-wise space-separated strings
        R_str = ' '.join([' '.join(map(str, row)) for row in R])
        t_str = ' '.join(map(str, t))
        
        current_result = {
            'scene_id': scene_id,
            'im_id': img_id,
            'obj_id': obj_id,
            'score': score,
            'R': R_str,
            't': t_str,
            'time': time,
            'final_iou': best_iou
        }
        
        # Keep track of the best result across all rotations
        # For rotated predictions, require at least 0.02 IOU improvement to select it
        min_improvement_threshold = 0.02 if rotation_angle != 0 else 0.0
        
        if best_iou > best_final_iou + min_improvement_threshold:
            best_final_iou = best_iou
            best_result = current_result
            print(f"{rotation_angle}° rotation prediction is currently the best (IoU: {best_iou:.4f})")
        else:
            if rotation_angle != 0 and best_iou > best_final_iou:
                print(f"{rotation_angle}° rotation has higher IoU ({best_iou:.4f} vs {best_final_iou:.4f}) but improvement ({best_iou - best_final_iou:.4f}) < threshold ({min_improvement_threshold})")
            else:
                print(f"{rotation_angle}° rotation prediction is worse than best (IoU: {best_iou:.4f} vs {best_final_iou:.4f})")
    
    # After comparing all predictions, return the best one
    # Remove the final_iou field before returning (not needed in CSV)
    if best_result:
        best_result.pop('final_iou', None)
    
    return best_result

def main():
    """Main function to evaluate PoseGAM on BOP dataset."""
    # Parse command-line arguments
    args = setup_args()

    cnos_masks = load_cnos_masks(args.BOP_dir, args.BOP_dataset_name)

    args.BOP_dir = os.path.join(args.BOP_dir, args.BOP_dataset_name)

    # Setup device and data type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load model
    model = load_model(device, model_path=args.model_path)

    # Set random seeds
    set_random_seeds(args.seed)

    # Load BOP sequences
    print(f"Loading BOP sequences from {args.BOP_dir}")

    test_json_path = os.path.join(args.BOP_query_dir.replace('tmp', args.BOP_dataset_name), 'test_targets_bop19.json')
    sequences = load_BOP_sequences(args.BOP_dir, args.min_num_images, test_json_path)
    
    if not sequences:
        print("No valid sequences found in the dataset directory!")
        return

    print(f"Found {len(sequences)} valid sequences")
    seq_names = sorted(list(sequences.keys()))

    seq_names = sorted(seq_names)

    # Split sequences by rank
    if args.total_ranks > 1:
        # Validate rank parameters
        if args.rank_id < 0 or args.rank_id >= args.total_ranks:
            raise ValueError(f"rank_id must be between 0 and {args.total_ranks-1}, got {args.rank_id}")
        
        # Split sequences among ranks
        total_sequences = len(seq_names)
        sequences_per_rank = (total_sequences + args.total_ranks - 1) // args.total_ranks  # Ceiling division
        start_idx = args.rank_id * sequences_per_rank
        end_idx = min(start_idx + sequences_per_rank, total_sequences)
        
        seq_names = seq_names[start_idx:end_idx]
        print(f"Rank {args.rank_id}/{args.total_ranks}: Processing sequences {start_idx} to {end_idx-1} (total: {len(seq_names)} sequences)")
    else:
        print(f"Single process mode: Processing all {len(seq_names)} sequences")

    print("Testing Sequences:")
    print(seq_names)
    print(f"Total sequences to process: {len(seq_names)}")

    # Prepare CSV file for results with rank-specific naming
    if args.total_ranks > 1:
        csv_filename = f"{args.output_name}_{args.BOP_dataset_name}_results_rank{args.rank_id}_of_{args.total_ranks}.csv"
    else:
        csv_filename = f"{args.output_name}_{args.BOP_dataset_name}_results.csv"
    csv_filepath = csv_filename
    
    # Read existing results to skip already processed sequences
    processed_results = {}  # Dict: {(scene_id, im_id, obj_id): count}
    if os.path.exists(csv_filepath):
        print(f"Found existing CSV file: {csv_filepath}")
        print("Reading existing results to resume processing...")
        with open(csv_filepath, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                key = (row['scene_id'], row['im_id'], row['obj_id'])
                processed_results[key] = processed_results.get(key, 0) + 1
        print(f"Found {len(processed_results)} unique sequences with {sum(processed_results.values())} total results already processed")
    else:
        # Write CSV header for new file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
        print(f"Created new CSV file: {csv_filepath}")
    
    print(f"Results will be saved to: {csv_filepath}")

    for i, seq_name in enumerate(seq_names):
        seq_frames = sequences[seq_name]
        print("-" * 50)
        print(f"Processing {seq_name} ({i+1}/{len(seq_names)})")

        known_camera_save_dir = None
        unknown_camera_save_dir = None

        inst_count = seq_frames.pop()
        
        # Parse sequence info to check against processed results
        scene_id = seq_name.split('/')[0].split('_')[0]
        img_id = seq_name.split('/')[0].split('_')[1]
        obj_id = seq_name.split('/')[1]
        
        # Check how many instances have already been processed for this sequence
        result_key = (scene_id, img_id, obj_id)
        already_processed_count = processed_results.get(result_key, 0)
        
        if already_processed_count >= inst_count:
            print(f"Skipping {seq_name} - all {inst_count} instances already processed")
            continue
        elif already_processed_count > 0:
            print(f"Resuming {seq_name} - {already_processed_count}/{inst_count} instances already processed, starting from inst_id={already_processed_count}")

        for inst_id in range(already_processed_count, inst_count):
            print(f"Processing instance {inst_id + 1}/{inst_count} for {seq_name}")
            result = process_sequence(
                model, seq_name, seq_frames, args.BOP_dir,
                args.min_num_images, args.num_frames, device, dtype, 
                args.whether_save, known_camera_save_dir=known_camera_save_dir, unknown_camera_save_dir=unknown_camera_save_dir, dataset_name=args.BOP_dataset_name, query_dir=args.BOP_query_dir, inst_id=inst_id, cnos_masks=cnos_masks
            )
            
            # Write result to CSV if successful
            if result is not None:
                with open(csv_filepath, 'a', newline='') as csvfile:
                    fieldnames = ['scene_id', 'im_id', 'obj_id', 'score', 'R', 't', 'time']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow(result)
                print(f"Result written to CSV: scene_id={result['scene_id']}, im_id={result['im_id']}, obj_id={result['obj_id']}, inst_id={inst_id}")
                
                # Update processed count in memory
                processed_results[result_key] = processed_results.get(result_key, 0) + 1
            else:
                print(f"Warning: Processing failed for {seq_name} inst_id={inst_id}")
    
    print("-" * 50)
    print(f"All results saved to: {csv_filepath}")

if __name__ == "__main__":
    main()