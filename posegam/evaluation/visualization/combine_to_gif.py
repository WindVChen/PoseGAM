"""
Given a root path, there are one or more sample folders (at any depth) with this structure:
- subfolder1
    query.png
    ... (other images ignored)
    - combined_results
        <method>_combined.png      # one per method found by 
        mask_combined.png

This script collects, for every image type, the matching image of each sample folder and writes
them out as a GIF (one GIF per image type), following the sorted order of the sample folders.
Nothing is hardcoded: the image types are discovered from the files that are actually there, so
any set of method names works. Sample folders are grouped by their parent directory, so a root
holding several datasets (e.g. comparison_results/ycbv, comparison_results/lmo) yields one
`gifs/` folder per dataset instead of mixing datasets into the same GIF.
"""

import os
from collections import defaultdict
from pathlib import Path
from PIL import Image
import argparse


QUERY_NAME = 'query.png'
COMBINED_SUBDIR = 'combined_results'
GIF_SUBDIR = 'gifs'


def create_gif_from_images(image_paths, output_path, duration=500):
    """
    Create a GIF from a list of image paths.

    Args:
        image_paths: List of paths to images
        output_path: Path where the GIF will be saved
        duration: Duration of each frame in milliseconds (default: 500ms)
    """
    frames = []
    for img_path in image_paths:
        try:
            frames.append(Image.open(img_path).convert('RGB'))
        except (OSError, ValueError) as e:
            print(f"Warning: Could not read {img_path}: {e}")

    if not frames:
        print(f"No readable images for {output_path}")
        return

    # Frames of different sizes would be cropped by PIL, so align them to the first frame.
    size = frames[0].size
    frames = [f if f.size == size else f.resize(size) for f in frames]

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    print(f"Created GIF ({len(frames)} frames): {output_path}")


def find_sample_dirs(root_path):
    """Find every sample folder under root_path, i.e. every folder holding a query.png."""
    return sorted(Path(dirpath) for dirpath, _, filenames in os.walk(root_path)
                  if QUERY_NAME in filenames)


def collect_frames(sample_dir):
    """Map image type -> image path for one sample: 'query' plus every combined_results/*.png."""
    frames = {Path(QUERY_NAME).stem: sample_dir / QUERY_NAME}
    for png in sorted((sample_dir / COMBINED_SUBDIR).glob('*.png')):
        frames[png.stem] = png
    return frames


def combine_group(sample_dirs, output_dir, duration):
    """Create one GIF per image type from the given sample folders."""
    # image type -> [path per sample folder that has it], keeping sample folder order
    frames_per_type = defaultdict(list)
    for sample_dir in sample_dirs:
        for image_type, image_path in collect_frames(sample_dir).items():
            frames_per_type[image_type].append(image_path)

    if not frames_per_type:
        print(f"No images found in {len(sample_dirs)} sample folder(s), skipping.")
        return

    # 'query' first (it is the raw input), then the rest alphabetically.
    image_types = sorted(frames_per_type, key=lambda t: (t != 'query', t))
    print(f"Found {len(image_types)} image types: {image_types}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for image_type in image_types:
        image_paths = frames_per_type[image_type]
        missing = len(sample_dirs) - len(image_paths)
        if missing:
            print(f"Warning: '{image_type}' is missing in {missing}/{len(sample_dirs)} sample folder(s)")
        create_gif_from_images(image_paths, output_dir / f"{image_type}.gif", duration)


def combine_to_gif(root_path, duration=500):
    """
    Process all sample folders under root_path and create GIF files for each image type.

    Args:
        root_path: Root directory containing sample folders (at any depth)
        duration: Duration of each frame in milliseconds
    """
    root_path = Path(root_path)

    if not root_path.is_dir():
        print(f"Error: root path does not exist or is not a directory: {root_path}")
        return

    sample_dirs = find_sample_dirs(root_path)
    if not sample_dirs:
        print(f"No sample folders (containing {QUERY_NAME}) found under {root_path}. "
              f"Did you run compare_methods_visualize.py / visualize_results.py first?")
        return

    # Group by parent so each dataset (or flat root) gets its own set of GIFs.
    groups = defaultdict(list)
    for sample_dir in sample_dirs:
        groups[sample_dir.parent].append(sample_dir)

    for parent, dirs in sorted(groups.items()):
        print(f"\n{'='*80}")
        print(f"{parent}: {len(dirs)} sample folder(s)")
        print(f"{'='*80}")
        combine_group(dirs, parent / GIF_SUBDIR, duration)
        print(f"GIFs saved to: {parent / GIF_SUBDIR}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine images from sample folders into GIF files')
    parser.add_argument('--root_path', type=str, default='./comparison_results',
                        help='Root directory whose sample folders each hold query.png + combined_results/ '
                             '(i.e. the output of visualize_results.py); nested layouts such as '
                             '<root>/<dataset>/<scene_im>/ are supported')
    parser.add_argument('--duration', type=int, default=1000,
                        help='Duration of each frame in milliseconds (default: 1000)')

    args = parser.parse_args()

    combine_to_gif(args.root_path, args.duration)
