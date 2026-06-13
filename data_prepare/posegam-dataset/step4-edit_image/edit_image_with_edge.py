#!/usr/bin/env python3
"""
Image editing script using FLUX.1-Canny-dev model.

This script processes images from the ObjaverseXL_Toys4k dataset, specifically:
1. Filters images with camera pitch between 15-60 degrees (above object)
2. Filters images with front-facing camera yaw (not backside views)
3. Checks object visibility using depth information
4. Applies Canny edge detection and uses FLUX.1-Canny-dev for image editing
5. Applies background masking using original RGBA alpha channel
6. Saves edited images and updated transforms.json

Features:
- Pitch filtering: Only cameras above object (positive elevation)
- Yaw filtering: Only front-facing cameras (configurable angular range)
- Background masking: Uses original image alpha channel to mask out background
- Debug mode: Saves original and unmasked edited images for comparison
- Parallel processing: Supports multi-GPU processing with rank-based distribution
- Resume functionality: Automatically skips completed SHA directories to resume interrupted processing
"""

import os
import json
import random
import ast
from typing import Dict, List

import cv2
import numpy as np
import torch
from PIL import Image
from controlnet_aux import CannyDetector
from diffusers import FluxControlPipeline
import pandas as pd

# Global parameters for processing
PITCH_MIN = 15.0
PITCH_MAX = 60.0
FRONT_ANGLE_RANGE = 120.0  # Front-facing angular range in degrees (±60° from front)
FLUX_PARAMS = {
    'num_inference_steps': 28,
    'guidance_scale': 30.0,
}


def load_metadata_csv(csv_path: str) -> Dict[str, List[str]]:
    """Load captions from metadata.csv file."""
    metadata = {}
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        sha256 = row['sha256']
        captions_str = row['captions']
        
        # Parse the captions string (it's a list in string format)
        try:
            captions = ast.literal_eval(captions_str)
            metadata[sha256] = captions
        except:
            print(f"Warning: Could not parse captions for {sha256}")
            metadata[sha256] = ["A 3D object"]
    
    return metadata


def calculate_pitch_from_transform_matrix(transform_matrix: np.ndarray) -> float:
    """
    Calculate pitch angle from camera transform matrix.
    
    In the Blender rendering setup:
    - Object is at origin (0, 0, 0) 
    - Camera is positioned using spherical coordinates (yaw, pitch, radius)
    - Positive pitch means camera is above the object (looking down)
    - Negative pitch means camera is below the object (looking up)
    
    Args:
        transform_matrix: 4x4 camera transform matrix from Blender
    
    Returns:
        Pitch angle in degrees (positive = above object, negative = below object)
    """
    # Extract camera position from transform matrix
    camera_position = transform_matrix[:3, 3]
    
    # Calculate distance from origin
    distance = np.linalg.norm(camera_position)
    
    if distance == 0:
        return 0.0
    
    # Pitch is the elevation angle: arcsin(z / distance)
    # Positive Z means camera is above the object
    pitch_rad = np.arcsin(camera_position[2] / distance)
    pitch_deg = np.degrees(pitch_rad)
    
    return pitch_deg


def calculate_yaw_from_transform_matrix(transform_matrix: np.ndarray) -> float:
    """
    Calculate yaw angle from camera transform matrix.
    
    In the Blender rendering setup:
    - Yaw is the azimuthal angle (rotation around Z-axis)
    - Front axis is directed along negative Y
    - Yaw = 0° means camera is on negative Y-axis (front)
    - Yaw = 90° means camera is on positive X-axis (right side)
    - Yaw = 180° means camera is on positive Y-axis (backside)
    - Yaw = 270° means camera is on negative X-axis (left side)
    
    Args:
        transform_matrix: 4x4 camera transform matrix from Blender
    
    Returns:
        Yaw angle in degrees (0-360°), where 0° is front (negative Y direction)
    """
    # Extract camera position from transform matrix
    camera_position = transform_matrix[:3, 3]
    
    # Calculate yaw angle relative to negative Y axis (front direction)
    # We use atan2(-y, x) to make negative Y the reference (0°)
    yaw_rad = np.arctan2(camera_position[0], -camera_position[1])
    yaw_deg = np.degrees(yaw_rad)
    
    return yaw_deg


def is_camera_front_facing(yaw_deg: float, front_angle_range: float = 120.0) -> bool:
    """
    Check if camera is positioned in front of the object based on yaw angle.
    
    Args:
        yaw_deg: Yaw angle in degrees (0-360°), where 0° is front (negative Y direction)
        front_angle_range: Angular range for "front-facing" in degrees
                          (e.g., 120° means ±60° from front, covering 120° total)
    
    Returns:
        True if camera is front-facing, False if at backside
    """
    # Define front-facing range around yaw = 0° (negative Y-axis, front direction)
    # For example, if front_angle_range = 120°, we want yaw in [-60°, +60°]
    # which translates to [300°, 360°] ∪ [0°, 60°] in 0-360° range
    
    half_range = front_angle_range / 2
    
    # Check if yaw is within front-facing range
    if yaw_deg <= half_range and yaw_deg >= - half_range:
        return True
    
    return False


def check_object_visibility(depth_path: str) -> bool:
    """
    Check if the object is fully visible in the image using depth information.
    
    Args:
        depth_path: Path to depth image (_depth.png file)
        threshold: Minimum fraction of object that should be visible
    
    Returns:
        True if object is sufficiently visible, False otherwise
    """
    if not os.path.exists(depth_path):
        print(f"Warning: Depth file not found: {depth_path}")
        return False
    
    try:
        # Load depth map as 16-bit (similar to co3d.py)
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_map is None:
            print(f"Warning: Could not load depth image: {depth_path}")
            return False
        
        # Set the value of pure white pixels to 0 which are the background
        # (following the pattern in co3d.py)
        depth_map[depth_map == 65535] = 0
        
        # Check if object parts are cut off by image boundary
        # Object pixels have non-zero depth values
        valid_depth = depth_map > 0
        
        if np.sum(valid_depth) == 0:
            return False
        
        # Check if object touches image boundaries
        height, width = depth_map.shape
        
        # Check top and bottom edges
        top_edge = valid_depth[0, :]
        bottom_edge = valid_depth[-1, :]
        left_edge = valid_depth[:, 0]
        right_edge = valid_depth[:, -1]
        
        # If object pixels are present at edges, it might be cut off
        boundary_pixels = (np.sum(top_edge) + np.sum(bottom_edge) + 
                          np.sum(left_edge) + np.sum(right_edge))
        
        if boundary_pixels > 0:
            return False
        
        return True
        
    except Exception as e:
        print(f"Error processing depth file {depth_path}: {e}")
        return False


def apply_canny_edge_detection(image: Image.Image) -> Image.Image:
    """Apply Canny edge detection to the input image using controlnet_aux."""
    # Initialize the Canny detector
    processor = CannyDetector()
    
    # Apply Canny edge detection with parameters similar to the example
    canny_image = processor(
        image, 
        low_threshold=50, 
        high_threshold=200, 
        detect_resolution=1024, 
        image_resolution=image.size[0]  # Use original image resolution
    )
    
    return canny_image


def setup_flux_pipeline():
    """Initialize the FLUX.1-Canny-dev pipeline."""
    # Load the pipeline directly
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Canny-dev", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    return pipe


def edit_image_with_flux(pipe, image: Image.Image, caption: str, 
                        num_inference_steps: int = 28,
                        guidance_scale: float = 3.5,
                        return_canny: bool = False) -> Image.Image:
    """
    Edit image using FLUX.1-Canny-dev model.
    
    Args:
        pipe: FLUX pipeline
        image: Input image
        caption: Text prompt for editing
        num_inference_steps: Number of inference steps
        guidance_scale: Guidance scale for generation
        return_canny: If True, return both edited image and Canny image
    
    Returns:
        Edited image, or tuple of (edited_image, canny_image) if return_canny=True
    """
    # Apply Canny edge detection
    canny_image = apply_canny_edge_detection(image)
    
    # Generate edited image using the correct FluxControlPipeline interface
    # Randomly set ddim_step_inverse between 0.3 and 0.8
    ddim_step_inverse = random.uniform(0.3, 0.8)
    result = pipe(
        prompt=caption,
        control_image=canny_image,
        height=canny_image.height,
        width=canny_image.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        ddim_step_inverse=ddim_step_inverse,
        ddim_inverse_image=image,
    ).images[0]
    
    if return_canny:
        return result, canny_image
    return result


def apply_background_mask(edited_image: Image.Image, original_image: Image.Image) -> Image.Image:
    """
    Apply background mask to edited image using the alpha channel from original RGBA image.
    
    Args:
        edited_image: RGB edited image from FLUX
        original_image: Original RGBA image with alpha channel
    
    Returns:
        RGBA edited image with background masked out
    """
    # Convert edited image to RGBA
    edited_rgba = edited_image.convert("RGBA")
    
    # Extract alpha channel from original image
    if original_image.mode == "RGBA":
        alpha_channel = original_image.split()[-1]  # Get alpha channel
    else:
        # If original is not RGBA, create a mask based on background detection
        # Assuming white/near-white background
        rgb_image = original_image.convert("RGB")
        rgb_array = np.array(rgb_image)
        
        # Create mask where background is black or near-black
        # Background pixels typically have low RGB values
        background_mask = np.all(rgb_array < 10, axis=2)
        alpha_array = np.where(background_mask, 0, 255).astype(np.uint8)
        alpha_channel = Image.fromarray(alpha_array, mode='L')
    
    # Apply alpha channel to edited image
    edited_rgba.putalpha(alpha_channel)
    
    return edited_rgba


def is_sha_already_processed(output_sha_dir: str) -> bool:
    """
    Check if a SHA directory has already been fully processed.
    
    A SHA directory is considered processed if it has a valid transforms.json file
    in the output directory, since this file is only saved after all images have
    been successfully edited.
    
    Args:
        output_sha_dir: Path to the output SHA directory
    
    Returns:
        True if the SHA directory is already processed, False otherwise
    """
    transforms_output_path = os.path.join(output_sha_dir, "transforms.json")
    
    if not os.path.exists(transforms_output_path):
        return False
    
    # Check if the transforms.json file is valid JSON
    try:
        with open(transforms_output_path, 'r') as f:
            transforms_data = json.load(f)
        
        # Check if it has the expected structure (frames list)
        if 'frames' not in transforms_data:
            return False
        
        # Check if frames list is not empty
        if not transforms_data['frames']:
            return False
        
        return True
    except (json.JSONDecodeError, IOError) as e:
        # If JSON is invalid or file cannot be read, consider it not processed
        print(f"Warning: Invalid transforms.json found at {transforms_output_path}: {e}")
        return False


def process_dataset(input_dir: str, output_dir: str, metadata_csv: str, 
                   rank_size: int = 1, rank_id: int = 0, debug: bool = False, no_resume: bool = False):
    """
    Edit a rendered dataset folder with FLUX.1-Canny-dev.

    Args:
        input_dir: Path to the input renders directory (e.g. renders3 or renders3-camTrans)
        output_dir: Path to the output directory (e.g. renders3-edited or renders3-camTrans-edited)
        metadata_csv: Path to metadata.csv file
        rank_size: Number of parallel processes
        rank_id: Current process rank ID
        debug: If True, save original images for comparison
        no_resume: If True, disable resume functionality and reprocess all directories
    """
    # Load metadata
    print(f"[Rank {rank_id}] Loading metadata...")
    captions_dict = load_metadata_csv(metadata_csv)
    
    # Setup FLUX pipeline
    print(f"[Rank {rank_id}] Setting up FLUX pipeline...")
    pipe = setup_flux_pipeline()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all SHA directories
    all_sha_dirs = [d for d in os.listdir(input_dir) 
                    if os.path.isdir(os.path.join(input_dir, d))]
    
    # Filter SHA directories based on rank for parallel processing
    if rank_size > 1:
        sha_dirs = [sha_dir for i, sha_dir in enumerate(all_sha_dirs) 
                   if i % rank_size == rank_id]
        print(f"[Rank {rank_id}] Processing {len(sha_dirs)} out of {len(all_sha_dirs)} SHA directories")
    else:
        sha_dirs = all_sha_dirs
        print(f"Processing {len(sha_dirs)} SHA directories")
    
    processed_count = 0
    successful_count = 0
    skipped_count = 0
    
    for sha_dir in sha_dirs:
        processed_count += 1
        print(f"[Rank {rank_id}] Processing {sha_dir} ({processed_count}/{len(sha_dirs)})...")
        
        # Check if this SHA directory is already fully processed (resume functionality)
        output_sha_dir = os.path.join(output_dir, sha_dir)
        if not no_resume and is_sha_already_processed(output_sha_dir):
            skipped_count += 1
            print(f"[Rank {rank_id}] Skipping {sha_dir}: already processed (found valid transforms.json)")
            continue
        
        sha_path = os.path.join(input_dir, sha_dir)
        transforms_path = os.path.join(sha_path, "transforms.json")
        
        if not os.path.exists(transforms_path):
            print(f"[Rank {rank_id}] Warning: transforms.json not found in {sha_path}")
            continue
        
        # Load transforms.json
        try:
            with open(transforms_path, 'r') as f:
                transforms_data = json.load(f)
        except Exception as e:
            print(f"[Rank {rank_id}] Error loading transforms.json for {sha_dir}: {e}")
            continue
        
        # Get captions for this SHA
        if sha_dir not in captions_dict:
            print(f"[Rank {rank_id}] Warning: No captions found for {sha_dir}")
            continue
        
        captions = captions_dict[sha_dir]
        
        # Process each frame
        edited_frames = []
        os.makedirs(output_sha_dir, exist_ok=True)
        
        frame_count = 0
        edited_frame_count = 0
        
        for frame in transforms_data['frames']:
            frame_count += 1
            file_path = frame['file_path']
            transform_matrix = np.array(frame['transform_matrix'])
            
            # Calculate pitch and yaw
            try:
                pitch = calculate_pitch_from_transform_matrix(transform_matrix)
                yaw = calculate_yaw_from_transform_matrix(transform_matrix)
            except Exception as e:
                print(f"[Rank {rank_id}] Error calculating camera angles for {file_path}: {e}")
                continue
            
            # Check if pitch is in desired range (15-60 degrees above object)
            # Positive pitch means camera is above the object (looking down)
            if not (PITCH_MIN <= pitch <= PITCH_MAX):
                print(f"[Rank {rank_id}] Skipping {file_path}: pitch {pitch:.1f}° outside range [{PITCH_MIN}°, {PITCH_MAX}°]")
                continue
            
            # Check if camera is front-facing (not at backside of object)
            if not is_camera_front_facing(yaw, FRONT_ANGLE_RANGE):
                print(f"[Rank {rank_id}] Skipping {file_path}: yaw {yaw:.1f}° indicates backside view (not front-facing)")
                continue
            
            # Check object visibility using depth
            image_path = os.path.join(sha_path, file_path)
            depth_path = image_path.replace('.png', '_depth.png')
            
            if not os.path.exists(depth_path):
                print(f"[Rank {rank_id}] Warning: Depth file not found: {depth_path}")
                continue
            
            if not check_object_visibility(depth_path):
                print(f"[Rank {rank_id}] Skipping {file_path}: object not fully visible")
                continue
            
            if not os.path.exists(image_path):
                print(f"[Rank {rank_id}] Warning: Image not found: {image_path}")
                continue
            
            # Load and edit image
            try:
                # Load original image in RGBA format to preserve alpha channel
                original_image = Image.open(image_path)
                
                # Convert to RGB for FLUX processing (FLUX expects RGB input)
                rgb_image = original_image.convert("RGB")
                
                # Save original image for debugging if requested
                if debug:
                    original_debug_path = os.path.join(output_sha_dir, f"original_{file_path}")
                    original_image.save(original_debug_path)
                
                # Randomly select a caption
                selected_caption = random.choice(captions)
                
                # Edit image with FLUX (get Canny image too if debug mode)
                if debug:
                    edited_image, canny_image = edit_image_with_flux(
                        pipe, rgb_image, selected_caption,
                        num_inference_steps=FLUX_PARAMS['num_inference_steps'],
                        guidance_scale=FLUX_PARAMS['guidance_scale'],
                        return_canny=True
                    )
                    # Save Canny edge detection image for debugging
                    canny_debug_path = os.path.join(output_sha_dir, f"canny_{file_path}")
                    canny_image.save(canny_debug_path)
                else:
                    edited_image = edit_image_with_flux(
                        pipe, rgb_image, selected_caption,
                        num_inference_steps=FLUX_PARAMS['num_inference_steps'],
                        guidance_scale=FLUX_PARAMS['guidance_scale']
                    )
                
                # Apply background mask using original alpha channel
                masked_edited_image = apply_background_mask(edited_image, original_image)
                
                # Save edited image with background masked
                output_image_path = os.path.join(output_sha_dir, file_path)
                masked_edited_image.save(output_image_path)
                
                # Add frame to edited frames list
                edited_frames.append(frame)
                edited_frame_count += 1
                
                print(f"[Rank {rank_id}] Edited and saved: {file_path} (pitch: {pitch:.1f}°, yaw: {yaw:.1f}°, masked: True)")
                
            except Exception as e:
                print(f"[Rank {rank_id}] Error processing {image_path}: {e}")
                continue
        
        # Save updated transforms.json for this SHA directory
        if edited_frames:
            output_transforms = {
                "frames": edited_frames
            }
            
            # Copy other metadata if present (aabb, scale, offset, etc.)
            for key in transforms_data:
                if key != "frames":
                    output_transforms[key] = transforms_data[key]
            
            output_transforms_path = os.path.join(output_sha_dir, "transforms.json")
            with open(output_transforms_path, 'w') as f:
                json.dump(output_transforms, f, indent=2)
            
            successful_count += 1
            print(f"[Rank {rank_id}] Saved {len(edited_frames)}/{frame_count} edited frames for {sha_dir}")
        else:
            # Remove empty directory
            try:
                os.rmdir(output_sha_dir)
            except:
                pass
            print(f"[Rank {rank_id}] No valid frames found for {sha_dir}")
    
    print(f"[Rank {rank_id}] Processing completed! Successfully processed {successful_count}/{processed_count} SHA directories")
    print(f"[Rank {rank_id}] Summary: {successful_count} processed, {skipped_count} skipped (already completed), {processed_count - successful_count - skipped_count} failed")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Edit images using FLUX.1-Canny-dev model")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/ibex/tmp/TRELLIS-500K/Toys4k/renders3",
        help="Input directory containing the rendered images to edit (e.g. renders3 or renders3-camTrans)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ibex/tmp/TRELLIS-500K/Toys4k/renders3-edited",
        help="Output directory for edited images (e.g. renders3-edited or renders3-camTrans-edited)"
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="/ibex/tmp/TRELLIS-500K/Toys4k/metadata.csv",
        help="Path to metadata.csv file"
    )
    parser.add_argument(
        "--rank_size",
        type=int,
        default=1,
        help="Number of processes to run in parallel. Default is 1 (no parallel processing)"
    )
    parser.add_argument(
        "--rank_id",
        type=int,
        default=0,
        help="Rank ID for parallel processing. Default is 0 (no parallel processing)"
    )
    parser.add_argument(
        "--pitch_min",
        type=float,
        default=15.0,
        help="Minimum pitch angle in degrees (default: 15.0)"
    )
    parser.add_argument(
        "--pitch_max",
        type=float,
        default=60.0,
        help="Maximum pitch angle in degrees (default: 60.0)"
    )
    parser.add_argument(
        "--front_angle_range",
        type=float,
        default=120.0,
        help="Front-facing angular range in degrees (default: 120.0, meaning ±60° from front)"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=28,
        help="Number of inference steps for FLUX (default: 28)"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=30.0,
        help="Guidance scale for FLUX (default: 30.0)"
    )
    parser.add_argument(
        "--debug",
        action="store_false",
        help="Debug images (original + Canny edge) are saved by DEFAULT; pass --debug to turn them OFF"
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Disable resume functionality and force reprocessing of all directories"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.rank_id >= args.rank_size:
        print(f"Error: rank_id ({args.rank_id}) must be less than rank_size ({args.rank_size})")
        return
    
    if args.pitch_min >= args.pitch_max:
        print(f"Error: pitch_min ({args.pitch_min}) must be less than pitch_max ({args.pitch_max})")
        return
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        print("Please ensure the input renders directory exists.")
        return
    
    if not os.path.exists(args.metadata_csv):
        print(f"Error: Metadata file not found: {args.metadata_csv}")
        return
    
    # Print configuration
    print("=" * 60)
    print("Image Editing with FLUX.1-Canny-dev")
    print("=" * 60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata file: {args.metadata_csv}")
    print(f"Pitch range: {args.pitch_min}° to {args.pitch_max}°")
    print(f"Front-facing range: {args.front_angle_range}° (±{args.front_angle_range/2:.1f}° from front)")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    print(f"Resume mode: {'Disabled (force reprocess)' if args.no_resume else 'Enabled (skip completed)'}")
    print(f"Parallel processing: rank {args.rank_id} of {args.rank_size}")
    print(f"FLUX parameters:")
    print(f"  - Inference steps: {args.num_inference_steps}")
    print(f"  - Guidance scale: {args.guidance_scale}")
    print(f"Background masking: Enabled (using RGBA alpha channel)")
    print("=" * 60)
    
    # Update global parameters
    global PITCH_MIN, PITCH_MAX, FRONT_ANGLE_RANGE, FLUX_PARAMS
    PITCH_MIN = args.pitch_min
    PITCH_MAX = args.pitch_max
    FRONT_ANGLE_RANGE = args.front_angle_range
    FLUX_PARAMS = {
        'num_inference_steps': args.num_inference_steps,
        'guidance_scale': args.guidance_scale
    }
    
    # Process the dataset
    process_dataset(args.input_dir, args.output_dir, args.metadata_csv, 
                   args.rank_size, args.rank_id, args.debug, args.no_resume)
    
    print(f"[Rank {args.rank_id}] Processing completed!")


if __name__ == "__main__":
    main()