"""
Sample random scene_id+im_id combinations and visualize rendering results from different methods.
For each sampled combination, save query image, mask, and renderings from all methods.
"""
import os
import json
import argparse
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
from pycocotools import mask as maskUtils
import trimesh
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import nvdiffrast.torch as dr


def load_mesh_with_textures(mesh_path: str, flip_uv_u: bool = False, flip_uv_v: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load mesh from file and return vertices, faces, UV coordinates, texture, and vertex colors as tensors."""
    scene = trimesh.load(mesh_path)
    
    if isinstance(scene, trimesh.Scene):
        meshes = [geometry for geometry in scene.geometry.values() if isinstance(geometry, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No meshes found in scene: {mesh_path}")
        mesh = meshes[0]
    else:
        mesh = scene
    
    texture_image = None
    if hasattr(mesh.visual, 'material') and mesh.visual.material is not None:
        material = mesh.visual.material
        if hasattr(material, 'baseColorTexture') and material.baseColorTexture is not None:
            texture_image = material.baseColorTexture
        elif hasattr(material, 'image') and material.image is not None:
            texture_image = material.image
    
    vertices = torch.from_numpy(mesh.vertices).float()
    faces = torch.from_numpy(mesh.faces).int()
    
    uv_coords = None
    if hasattr(mesh.visual, 'uv') and mesh.visual.uv is not None:
        uv_coords = torch.from_numpy(mesh.visual.uv).float()
        if flip_uv_u:
            uv_coords[:, 0] = 1.0 - uv_coords[:, 0]
        if flip_uv_v:
            uv_coords[:, 1] = 1.0 - uv_coords[:, 1]
    
    texture = None
    if texture_image is not None:
        if isinstance(texture_image, Image.Image):
            texture_np = np.array(texture_image)
        else:
            texture_np = np.array(Image.open(texture_image))
        
        if len(texture_np.shape) == 2:
            texture_np = np.stack([texture_np] * 3, axis=-1)
        elif texture_np.shape[2] == 4:
            texture_np = texture_np[:, :, :3]
        
        texture = torch.from_numpy(texture_np).float() / 255.0
    
    vertex_colors = None
    if hasattr(mesh.visual, 'material') and texture is not None and uv_coords is not None:
        colors = mesh.visual.material.to_color(uv=np.clip(mesh.visual.uv, 0, 1 - 1e-6))
        if colors.shape[1] >= 3:
            vertex_colors = torch.from_numpy(colors[:, :3]).float() / 255.0
    
    return vertices, faces, uv_coords, texture, vertex_colors


# Note: normalize_mesh and blender_to_opencv_camera_matrix functions removed
# as they are not needed for BOP dataset rendering where R,t are already in camera coordinates


def intrinsics_to_projection(intrinsics: torch.Tensor, near: float = 0.1, far: float = 1000.0,
                            image_width: int = 1, image_height: int = 1) -> torch.Tensor:
    """OpenCV intrinsics to OpenGL perspective matrix."""
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    fx_normalized = fx / (image_width / 2.0)
    fy_normalized = fy / (image_height / 2.0)
    cx_normalized = (2.0 * cx / image_width) - 1.0
    cy_normalized = 1 - (2.0 * cy / image_height)
    
    ret = torch.zeros((4, 4), dtype=intrinsics.dtype, device=intrinsics.device)
    ret[0, 0] = fx_normalized
    ret[1, 1] = fy_normalized
    ret[0, 2] = cx_normalized
    ret[1, 2] = -cy_normalized
    ret[2, 2] = far / (far - near)
    ret[2, 3] = near * far / (near - far)
    ret[3, 2] = 1.
    return ret


def render_mesh(vertices: torch.Tensor, faces: torch.Tensor, camera_matrix: torch.Tensor,
                projection_matrix: torch.Tensor, uv_coords: Optional[torch.Tensor] = None,
                texture: Optional[torch.Tensor] = None, vertex_colors: Optional[torch.Tensor] = None,
                shading_mode: str = 'texture', resolution: int = 512, device: str = 'cuda',
                width: Optional[int] = None, height: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render mesh using nvdiffrast."""
    vertices = vertices.to(device)
    faces = faces.to(device).int()
    camera_matrix = camera_matrix.to(device)
    projection_matrix = projection_matrix.to(device)
    
    vertices = vertices.unsqueeze(0)
    faces = faces.unsqueeze(0)
    
    vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
    view_matrix = camera_matrix
    vertices_cam = torch.matmul(vertices_homo, view_matrix.T)
    vertices_clip = torch.matmul(vertices_cam, projection_matrix.T)
    
    use_texture_sampling = False
    if shading_mode == 'texture' and uv_coords is not None and texture is not None:
        use_texture_sampling = True
        uv_coords = uv_coords.to(device)
        uv_coords = torch.clamp(uv_coords, 0.0, 1.0)
        colors = uv_coords.unsqueeze(0)
    elif shading_mode == 'vertex_color' and vertex_colors is not None:
        vertex_colors = vertex_colors.to(device)
        colors = vertex_colors.unsqueeze(0)
        colors = torch.cat([colors, torch.ones(1, colors.shape[1], 1, device=device)], dim=-1)
    elif shading_mode == 'normal':
        face_vertices = vertices[0, faces[0]]
        v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
        face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
        face_normals = torch.nn.functional.normalize(face_normals, p=2, dim=1)
        
        num_vertices = vertices.shape[1]
        vertex_normals = torch.zeros(num_vertices, 3, device=device)
        vertex_count = torch.zeros(num_vertices, device=device)
        
        face_indices = faces[0].flatten().long()
        face_normals_expanded = face_normals.repeat_interleave(3, dim=0)
        
        vertex_normals.scatter_add_(0, face_indices.unsqueeze(1).expand(-1, 3), face_normals_expanded)
        vertex_count.scatter_add_(0, face_indices, torch.ones_like(face_indices, dtype=torch.float32))
        
        valid_vertices = vertex_count > 0
        vertex_normals[valid_vertices] /= vertex_count[valid_vertices].unsqueeze(1)
        vertex_normals = torch.nn.functional.normalize(vertex_normals, p=2, dim=1)
        
        vertex_colors_rgb = (vertex_normals + 1.0) * 0.5
        vertex_colors_rgb = torch.clamp(vertex_colors_rgb, 0.0, 1.0)
        
        vertex_colors = vertex_colors_rgb.unsqueeze(0)
        colors = torch.cat([vertex_colors, torch.ones(1, vertex_colors.shape[1], 1, device=device)], dim=-1)
    else:
        raise ValueError(f"Unknown shading mode: {shading_mode}")
    
    glctx = dr.RasterizeGLContext()
    
    actual_width = width if width is not None else resolution
    actual_height = height if height is not None else resolution
    
    rast_out, rast_db = dr.rasterize(glctx, vertices_clip, faces[0], resolution=[actual_height, actual_width], grad_db=True)
    
    depth = rast_out[..., 2:3]
    triangle_id = rast_out[..., 3:4]
    
    if use_texture_sampling:
        uv_interpolated, uv_dr = dr.interpolate(colors, rast_out, faces[0], rast_db, diff_attrs='all')
        texture = texture.to(device)
        if len(texture.shape) == 3:
            texture_batch = texture.unsqueeze(0)
        else:
            texture_batch = texture
        rgb_image = dr.texture(texture_batch, uv_interpolated, uv_dr)
    else:
        rgb_image, _ = dr.interpolate(colors, rast_out, faces[0])
    
    if use_texture_sampling:
        rgb_image = dr.antialias(rgb_image, rast_out, vertices_clip, faces[0])
        mask = dr.antialias((rast_out[..., -1:] > 0).float(), rast_out, vertices_clip, faces[0])
        rgb_image = torch.where(mask > 0, rgb_image, torch.zeros_like(rgb_image))
        rgb_image = torch.cat([rgb_image, torch.ones_like(rgb_image[..., :1])], dim=-1)
    else:
        rgb_image = dr.antialias(rgb_image, rast_out, vertices_clip, faces[0])
    
    background_mask = (triangle_id == 0).squeeze(-1)
    rgb_image = rgb_image.squeeze(0)
    rgb_image[background_mask.squeeze(0)] = torch.tensor([0.0, 0.0, 0.0, 0.0], device=device)
    
    depth_image = depth.squeeze()
    depth_image[background_mask.squeeze(0)] = 0.0
    
    return rgb_image, depth_image


def save_image(image: torch.Tensor, filepath: str):
    """Save tensor image to file with transparent background."""
    image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    
    if image_np.shape[2] == 4:
        # Keep alpha channel as is for transparency
        # Where alpha is 0 (background), keep it transparent
        image_pil = Image.fromarray(image_np, mode='RGBA')
    else:
        # If no alpha channel, add one (shouldn't happen with our rendering)
        image_pil = Image.fromarray(image_np)
    
    image_pil.save(filepath)


def parse_rotation_matrix(r_str: str) -> np.ndarray:
    """Parse rotation matrix from CSV string format."""
    r_values = [float(x) for x in r_str.split()]
    return np.array(r_values).reshape(3, 3)


def parse_translation_vector(t_str: str) -> np.ndarray:
    """Parse translation vector from CSV string format."""
    t_values = [float(x) for x in t_str.split()]
    return np.array(t_values)


def create_transform_matrix(R: np.ndarray, t: np.ndarray) -> List[List[float]]:
    """Create 4x4 transform matrix from R and t."""
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    return transform.tolist()


def get_dataset_config(dataset_name: str) -> Dict:
    """Get dataset-specific configuration."""
    configs = {
        'ycbv': {
            'image_ext': '.rgb.png',
            'use_primesense': False,
            'is_gray': False,
        },
        'lmo': {
            'image_ext': '.rgb.png',
            'use_primesense': False,
            'is_gray': False,
        },
        'tudl': {
            'image_ext': '.rgb.png',
            'use_primesense': False,
            'is_gray': False,
        },
        'icbin': {
            'image_ext': '.rgb.png',
            'use_primesense': False,
            'is_gray': False,
        },
        'tless': {
            'image_ext': '.rgb.png',
            'use_primesense': True,
            'is_gray': False,
        },
    }
    return configs.get(dataset_name, configs['ycbv'])


def sample_scene_im_combinations(csv_files: Dict[str, str], n_samples: int, seed: int = 42) -> List[Tuple[str, str, List[Tuple[str, int]]]]:
    """Sample random N scene_id+im_id combinations that exist across all methods.
    
    Returns:
        List of tuples (scene_id, im_id, [(obj_id, instance_count), ...]) 
        where instance_count is the number of instances of that obj_id
    """
    random.seed(seed)
    
    # Load all CSV files
    dataframes = {}
    for method_name, csv_path in csv_files.items():
        df = pd.read_csv(csv_path)
        dataframes[method_name] = df
        print(f"Loaded {method_name}: {len(df)} rows")
    
    # Find common scene_id+im_id combinations across all methods
    common_scene_im = None
    
    for method_name, df in dataframes.items():
        scene_im_combinations = set(zip(df['scene_id'].astype(str).str.zfill(6), 
                                       df['im_id'].astype(str).str.zfill(6)))
        if common_scene_im is None:
            common_scene_im = scene_im_combinations
        else:
            common_scene_im = common_scene_im.intersection(scene_im_combinations)
    
    print(f"Found {len(common_scene_im)} common scene+image combinations across all methods")
    
    # For each common scene+image, find object instances that exist in ALL methods
    scene_im_with_objs = []
    for scene_id, im_id in common_scene_im:
        # For each method, count instances per obj_id
        method_obj_counts = {}
        for method_name, df in dataframes.items():
            mask = (df['scene_id'].astype(str).str.zfill(6) == scene_id) & \
                   (df['im_id'].astype(str).str.zfill(6) == im_id)
            obj_ids = df[mask]['obj_id'].astype(str).tolist()
            # Count instances per object ID
            obj_count = {}
            for obj_id in obj_ids:
                obj_count[obj_id] = obj_count.get(obj_id, 0) + 1
            method_obj_counts[method_name] = obj_count
        
        # Find objects with matching instance counts across ALL methods
        if not method_obj_counts:
            continue
        
        # Get all unique obj_ids
        all_obj_ids = set()
        for counts in method_obj_counts.values():
            all_obj_ids.update(counts.keys())
        
        # For each obj_id, find the minimum instance count across all methods
        common_obj_instances = []
        for obj_id in all_obj_ids:
            instance_counts = [method_obj_counts[method].get(obj_id, 0) 
                              for method in method_obj_counts.keys()]
            min_instances = min(instance_counts)
            if min_instances > 0:
                common_obj_instances.append((obj_id, min_instances))
        
        if common_obj_instances:
            scene_im_with_objs.append((scene_id, im_id, sorted(common_obj_instances)))
    
    print(f"Found {len(scene_im_with_objs)} scene+image combinations with common objects")

    # Sample N combinations
    sampled = random.sample(scene_im_with_objs, min(n_samples, len(scene_im_with_objs)))
    print(f"Sampled {len(sampled)} scene+image combinations")

    return sampled


def render_for_pose(mesh_path: str, R: np.ndarray, t: np.ndarray, 
                   image_width: int, image_height: int, fx: float, fy: float,
                   cx: float, cy: float,
                   obj_id: str, dataset_name: str, tmp_dir: str,
                   device: str = 'cuda', shading_mode: str = 'texture') -> torch.Tensor:
    """Render mesh for a given pose."""
    # Load mesh
    vertices, faces, uv_coords, texture, vertex_colors = load_mesh_with_textures(mesh_path)
    
    # R and t from BOP CSV represent object pose in camera coordinates (object-to-camera)
    # We need to create the transformation matrix properly for rendering
    
    # Create camera intrinsics matrix using actual principal point from camera parameters
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32)
    
    # Create object-to-camera transformation matrix (4x4)
    obj_to_cam = torch.eye(4, dtype=torch.float32)
    obj_to_cam[:3, :3] = torch.from_numpy(R).float()

    # Load model info to get scaling information
    models_info_path = os.path.join(tmp_dir.replace('tmp', dataset_name), 'models' if dataset_name != 'tless' else 'models_cad', 'models_info.json')
    if os.path.exists(models_info_path):
        model_info = json.load(open(models_info_path, 'r'))[str(obj_id)]
        model_initial_size = [model_info['size_x'], model_info['size_y'], model_info['size_z']]
        model_scale = max(model_initial_size)
        # Apply scaling to translation - convert from model scale to normalized scale
        obj_to_cam[:3, 3] = torch.from_numpy(t).float() / model_scale * 2.1
    else:
        # Fallback: use translation as-is if model info not available
        obj_to_cam[:3, 3] = torch.from_numpy(t).float()
    
    # The view matrix for rendering is the object-to-camera transform
    view_matrix = obj_to_cam

    z_up_to_y_up = torch.tensor([
        [1,  0,  0,  0],
        [0,  0,  1,  0],  # Y <- Z
        [0, -1,  0,  0],  # Z <- -Y
        [0,  0,  0,  1]
    ], dtype=torch.float32)

    view_matrix = view_matrix @ z_up_to_y_up.inverse()
    
    # Create projection matrix
    projection_matrix = intrinsics_to_projection(
        intrinsics, near=0.1, far=100.0,
        image_width=image_width, image_height=image_height
    )
    
    # Render
    rgb_image, depth_image = render_mesh(
        vertices, faces, view_matrix, projection_matrix,
        uv_coords, texture, vertex_colors,
        shading_mode=shading_mode, device=device,
        width=image_width, height=image_height
    )
    
    return rgb_image


def process_combination(scene_id: str, im_id: str, obj_instances: List[Tuple[str, int]],
                       dataset_name: str, methods_data: Dict,
                       tmp_dir: str, mesh_root: str, output_dir: str,
                       device: str = 'cuda', shading_mode: str = 'texture'):
    """Process a single scene_id+im_id combination with all its objects.
    
    Args:
        obj_instances: List of tuples (obj_id, instance_count) for objects in this scene+image
    """
    config = get_dataset_config(dataset_name)
    
    # Create output folder for this combination
    combo_dir = os.path.join(output_dir, f"{scene_id}_{im_id}")
    os.makedirs(combo_dir, exist_ok=True)
    
    # Get query image path - format: {scene_id:6d}_{im_id:6d}.rgb.png
    test_folder = 'test' + ('_primesense' if config['use_primesense'] else '')
    query_image_path = os.path.join(
        tmp_dir, f"{dataset_name}_image_wise", test_folder,
        f"{scene_id}_{im_id}{config['image_ext']}"
    )

    camera_path = os.path.join(
        tmp_dir, f"{dataset_name}_image_wise", test_folder,
        f"{scene_id}_{im_id}.camera.json"
    )
    
    camera_info = json.load(open(camera_path, 'r'))["cam_K"]
    fx, fy = (camera_info[0], camera_info[4])
    cx, cy = (camera_info[2], camera_info[5])  # Principal point coordinates
    
    # Copy query image
    if os.path.exists(query_image_path):
        query_img = Image.open(query_image_path)
        query_img.save(os.path.join(combo_dir, "query.png"))
        image_width, image_height = query_img.size
    else:
        print(f"Warning: Query image not found: {query_image_path}")
        return
    
    # Load ground truth to get all object instances in this scene+image
    gt_path = os.path.join(
        tmp_dir, f"{dataset_name}_image_wise", test_folder,
        f"{scene_id}_{im_id}.gt.json"
    )
    
    # Load GT to create object ID to indices mapping (support multiple instances)
    obj_id_to_indices = {}
    if os.path.exists(gt_path):
        gt_data = json.load(open(gt_path, 'r'))
        for idx, gt_item in enumerate(gt_data):
            obj_id_str = str(gt_item["obj_id"])
            if obj_id_str not in obj_id_to_indices:
                obj_id_to_indices[obj_id_str] = []
            obj_id_to_indices[obj_id_str].append(idx)
    
    # Load masks for all objects we're processing
    mask_path = os.path.join(
        tmp_dir, f"{dataset_name}_image_wise", test_folder,
        f"{scene_id}_{im_id}.mask_visib.json"
    )
    
    # Load and save individual mask for each object instance
    if os.path.exists(mask_path):
        mask_data = json.load(open(mask_path, 'r'))
        # Save individual mask for each object instance
        for obj_id, instance_count in obj_instances:
            if obj_id in obj_id_to_indices:
                indices = obj_id_to_indices[obj_id][:instance_count]  # Take only the instances we need
                for inst_idx, gt_idx in enumerate(indices):
                    if str(gt_idx) in mask_data:
                        mask_info = mask_data[str(gt_idx)]
                        binary_mask = maskUtils.decode(maskUtils.frPyObjects(
                            mask_info, mask_info["size"][0], mask_info["size"][1]
                        ))
                        # Create RGBA image with transparency
                        mask_rgba = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
                        # Set white color where mask is 1, and alpha channel based on mask
                        mask_rgba[..., :3] = 255  # RGB channels to white
                        mask_rgba[..., 3] = binary_mask.astype(np.uint8) * 255  # Alpha channel from mask
                        mask_img = Image.fromarray(mask_rgba, mode='RGBA')
                        # Save with instance index if multiple instances
                        inst_suffix = f"_inst{inst_idx}" if instance_count > 1 else ""
                        mask_img.save(os.path.join(combo_dir, f"obj{obj_id.zfill(6)}{inst_suffix}_mask.png"))
    else:
        print(f"Warning: Mask not found: {mask_path}")
    
    # Process each object instance
    for obj_id, instance_count in obj_instances:
        print(f"  Processing object {obj_id} with {instance_count} instance(s)")
        
        # Render for each method
        mesh_path = os.path.join(mesh_root, dataset_name, f"obj_{obj_id.zfill(6)}", "mesh.glb")
        
        if not os.path.exists(mesh_path):
            print(f"Warning: Mesh not found: {mesh_path}")
            continue
        
        for method_name, df in methods_data.items():
            # Find all rows for this combination and object
            rows = df[(df['scene_id'].astype(str).str.zfill(6) == scene_id) & 
                     (df['im_id'].astype(str).str.zfill(6) == im_id) &
                     (df['obj_id'].astype(str) == obj_id)]
            
            if len(rows) == 0:
                print(f"Warning: No data for {method_name} in {scene_id}_{im_id}_{obj_id}")
                continue
            
            # Process each instance (limit to instance_count)
            for inst_idx in range(min(instance_count, len(rows))):
                row = rows.iloc[inst_idx]
                
                # Parse R and t
                R = parse_rotation_matrix(row['R'])
                t = parse_translation_vector(row['t'])
                
                # Render
                rgb_image = render_for_pose(
                    mesh_path, R, t, image_width, image_height, fx, fy, cx, cy,
                    obj_id, dataset_name, tmp_dir,
                    device=device, shading_mode=shading_mode
                )
                
                # Save rendered image with object ID and instance index
                inst_suffix = f"_inst{inst_idx}" if instance_count > 1 else ""
                save_image(rgb_image, os.path.join(combo_dir, f"obj{obj_id.zfill(6)}{inst_suffix}_{method_name}.png"))


def main():
    parser = argparse.ArgumentParser(description='Visualize BOP method comparisons')
    parser.add_argument('--compare_dir', type=str, default='./compare-draw',
                       help='Root directory of per-method result CSVs, laid out as '
                            '<compare_dir>/<dataset>/<method>.csv (e.g. gt.csv, ours.csv, gigapose.csv, ...)')
    parser.add_argument('--datasets', type=str, nargs='+', default=['ycbv'],
                       help='Dataset names to process')
    parser.add_argument('--n_samples', type=int, default=50,
                       help='Number of random samples per dataset')
    parser.add_argument('--tmp_dir', type=str,
                       default='/path/to/gigapose/gigaPose_datasets/datasets/tmp',
                       help='gigapose image_wise directory (provides BOP query images + intrinsics)')
    parser.add_argument('--mesh_root', type=str, default='/ibex/tmp/TRELLIS-500K/BOP-data',
                       help='BOP_dir with the watertight meshes (<mesh_root>/<dataset>/obj_<id>/mesh.glb)')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Output directory for visualization results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--shading_mode', type=str, default='normal',
                       choices=['texture', 'vertex_color', 'normal'],
                       help='Shading mode for rendering')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each dataset
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_dir = os.path.join(args.compare_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            print(f"Warning: Dataset directory not found: {dataset_dir}")
            continue
        
        # Find all method CSV files
        method_files = {}
        for file in os.listdir(dataset_dir):
            if file.endswith('.csv'):
                method_name = file.replace('.csv', '')
                method_files[method_name] = os.path.join(dataset_dir, file)
        
        print(f"Found methods: {list(method_files.keys())}")
        
        # Sample combinations
        sampled_combinations = sample_scene_im_combinations(
            method_files, args.n_samples, args.seed
        )
        
        # Load all method dataframes
        methods_data = {}
        for method_name, csv_path in method_files.items():
            methods_data[method_name] = pd.read_csv(csv_path)
        
        # Create dataset output directory
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Process each combination
        for scene_id, im_id, obj_instances in tqdm(sampled_combinations, desc=f"Processing {dataset_name}"):
            total_instances = sum(count for _, count in obj_instances)
            print(f"\nProcessing {scene_id}_{im_id} with {len(obj_instances)} unique objects, {total_instances} total instances")
            process_combination(
                scene_id, im_id, obj_instances,
                dataset_name, methods_data,
                args.tmp_dir, args.mesh_root, dataset_output_dir,
                args.device, args.shading_mode
            )
        
        print(f"Completed {dataset_name}: Results saved to {dataset_output_dir}")
    
    print(f"\n{'='*60}")
    print(f"All processing complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
