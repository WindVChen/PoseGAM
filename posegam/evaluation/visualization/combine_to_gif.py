"""
Given a root path, there are multiple subfolders each has this structure:
- subfolder1
    query.png
    ... (other images ignore)
    - combined_results
        foundpose_combined.png
        genflow_combined.png
        gigapose_combined.png
        gt_combined.png
        mask_combined.png
        megapose_combined.png
        ours_combined.png
        vggt_combined.png

This script goes through each subfolder and combines these images into gif files respectively,
according to the order of subfolders.
"""

import os
from pathlib import Path
from PIL import Image
import argparse


def create_gif_from_images(image_paths, output_path, duration=500):
    """
    Create a GIF from a list of image paths.
    
    Args:
        image_paths: List of paths to images
        output_path: Path where the GIF will be saved
        duration: Duration of each frame in milliseconds (default: 500ms)
    """
    images = []
    for img_path in image_paths:
        if os.path.exists(img_path):
            images.append(Image.open(img_path))
        else:
            print(f"Warning: Image not found: {img_path}")
    
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Created GIF: {output_path}")
    else:
        print(f"No images found for {output_path}")


def combine_to_gif(root_path, duration=500):
    """
    Process all subfolders in root_path and create GIF files for each image type.
    
    Args:
        root_path: Root directory containing subfolders
        duration: Duration of each frame in milliseconds
    """
    root_path = Path(root_path)
    
    # Get all subfolders, sorted by name
    subfolders = sorted([d for d in root_path.iterdir() if d.is_dir()])
    
    if not subfolders:
        print("No subfolders found in the root path.")
        return
    
    # Image types to process
    image_types = [
        'query.png',
        'foundpose_combined.png',
        'genflow_combined.png',
        'gigapose_combined.png',
        'gt_combined.png',
        'mask_combined.png',
        'megapose_combined.png',
        'ours_combined.png',
        'vggt_combined.png'
    ]
    
    # Create output directory
    output_dir = root_path / 'gifs'
    output_dir.mkdir(exist_ok=True)
    
    # Process each image type
    for image_type in image_types:
        image_paths = []
        
        # Collect images from all subfolders
        for subfolder in subfolders:
            # query.png is in the subfolder root, others are in combined_results
            if image_type == 'query.png':
                image_path = subfolder / image_type
            else:
                combined_results = subfolder / 'combined_results'
                image_path = combined_results / image_type
            
            if image_path.exists():
                image_paths.append(image_path)
        
        # Create GIF if we have images
        if image_paths:
            gif_name = image_type.replace('.png', '.gif')
            output_path = output_dir / gif_name
            create_gif_from_images(image_paths, output_path, duration)
        else:
            print(f"No images found for {image_type}")
    
    print(f"\nAll GIFs saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine images from subfolders into GIF files')
    parser.add_argument('--root_path', type=str, default='./comparison_results',
                        help='Root directory whose subfolders each hold query.png + combined_results/ '
                             '(i.e. the output of visualize_results.py)')
    parser.add_argument('--duration', type=int, default=1000,
                        help='Duration of each frame in milliseconds (default: 1000)')
    
    args = parser.parse_args()
    
    combine_to_gif(args.root_path, args.duration)