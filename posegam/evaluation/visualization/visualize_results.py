import cv2
import numpy as np
from pathlib import Path
import os
import argparse
import re

# Predefined colors for objects (BGR format)
# Expanded color palette - each object name will deterministically get the same color
PREDEFINED_COLORS = [
    (255, 0, 0),      # Blue
    (0, 0, 255),      # Red
    (0, 255, 255),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 0),      # Green
    (255, 128, 0),    # Sky Blue
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Orange
    (255, 255, 0),    # Cyan
    (128, 255, 0),    # Light Green
    (255, 0, 128),    # Pink
    (0, 255, 128),    # Spring Green
    (192, 192, 192),  # Silver
    (128, 128, 0),    # Olive
    (128, 0, 128),    # Dark Purple
    (0, 128, 128),    # Teal
    (255, 165, 0),    # Light Orange
    (75, 0, 130),     # Indigo
    (238, 130, 238),  # Violet
    (255, 192, 203),  # Light Pink
    (64, 224, 208),   # Turquoise
    (255, 215, 0),    # Gold
    (218, 112, 214),  # Orchid
    (240, 128, 128),  # Light Coral
    (32, 178, 170),   # Light Sea Green
    (219, 112, 147),  # Pale Violet Red
    (255, 20, 147),   # Deep Pink
    (72, 61, 139),    # Dark Slate Blue
    (154, 205, 50),   # Yellow Green
    (255, 140, 0),    # Dark Orange
]

# Border thickness
BORDER_THICKNESS = 3

def get_color_for_object(obj_name):
    """Get a deterministic color for an object based on its name"""
    # Use hash of object name to get consistent color assignment
    hash_value = hash(obj_name)
    color_index = hash_value % len(PREDEFINED_COLORS)
    return PREDEFINED_COLORS[color_index]

def assign_colors_to_objects(object_names):
    """Assign consistent colors to objects based on their names"""
    color_map = {}
    for obj_name in object_names:
        color_map[obj_name] = get_color_for_object(obj_name)
    return color_map

def extract_foreground_mask(image):
    """Extract foreground mask from image using alpha channel"""
    # Check if image has alpha channel
    if image.shape[2] == 4:
        # Use alpha channel directly
        alpha = image[:, :, 3]
        # Create mask where alpha > 0
        mask = (alpha > 0).astype(np.uint8) * 255
    else:
        # Fallback: convert to grayscale and threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    return mask

def add_colored_border(image, mask, color, thickness=3):
    """Add a colored border around the mask on the image"""
    # Find contours of the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours with the specified color
    result = image.copy()
    cv2.drawContours(result, contours, -1, color, thickness)
    
    return result

def paste_foreground_with_border(base_image, overlay_image, color, thickness=3):
    """Paste the foreground of overlay_image onto base_image with a colored border"""
    # Convert overlay to BGR if it has alpha channel
    if len(overlay_image.shape) == 3 and overlay_image.shape[2] == 4:
        overlay_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_BGRA2BGR)
    else:
        overlay_bgr = overlay_image
    
    # Extract foreground mask
    mask = extract_foreground_mask(overlay_image)
    
    # Create a copy of base image
    result = base_image.copy()
    
    # Paste the overlay foreground onto the base image
    mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    foreground = cv2.bitwise_and(overlay_bgr, mask_3channel)
    background = cv2.bitwise_and(result, cv2.bitwise_not(mask_3channel))
    result = cv2.add(foreground, background)
    
    # Add colored border
    result = add_colored_border(result, mask, color, thickness)
    
    return result

def color_foreground_region(base_image, mask_image, color):
    """Color the foreground region with pure color and darken background"""
    # Extract foreground mask using alpha channel
    mask = extract_foreground_mask(mask_image)
    
    # Create a copy of base image
    result = base_image.copy().astype(np.float32)
    
    # Darken the background (multiply by 0.5)
    result = result * 0.5
    
    # Create mask for foreground (where mask > 0)
    mask_bool = mask > 0
    
    # Set foreground to pure color
    for i in range(3):  # BGR channels
        result[:, :, i][mask_bool] = color[i]
    
    return result.astype(np.uint8)

def process_directory(work_dir, output_dir, border_thickness):
    """Process a single directory containing query.png and object files"""
    # Load query image (with alpha channel if available)
    query_path = work_dir / "query.png"
    query_img = cv2.imread(str(query_path), cv2.IMREAD_UNCHANGED)
    
    if query_img is None:
        print(f"  Warning: Could not load query image from {query_path}")
        return False
    
    # Convert to BGR if it has alpha channel (for display purposes)
    if len(query_img.shape) == 3 and query_img.shape[2] == 4:
        query_img = cv2.cvtColor(query_img, cv2.COLOR_BGRA2BGR)
    
    print(f"  Query image loaded: {query_img.shape}")
    
    # Gather files and robustly parse filenames like:
    #  obj000001_foundpose.png
    #  obj000001_inst0_foundpose.png
    #  obj000001_inst1_foundpose.png
    files = os.listdir(work_dir)

    # Regex to capture object id and method name, allowing optional instance.
    # groups: 1 -> obj id (e.g. obj000001), 2 -> method name (e.g. foundpose)
    pattern = re.compile(r'^(obj\d+)(?:_inst\d+)?_(.+)\.png$', re.IGNORECASE)

    # Build mapping: object_id -> method -> [filenames]
    file_map = {}
    methods_set = set()
    for f in files:
        m = pattern.match(f)
        if not m:
            continue
        obj_id = m.group(1)
        method = m.group(2)
        methods_set.add(method)
        file_map.setdefault(obj_id, {}).setdefault(method, []).append(f)

    objects = sorted(file_map.keys())
    if not objects:
        print("  Warning: No object files found (expected files like obj000001_*.png)")
        return False

    print(f"  Found {len(objects)} objects: {objects}")

    # Assign colors to all objects (deterministic based on object name)
    object_colors = assign_colors_to_objects(objects)

    print(f"  Color assignments:")
    for obj_name, color in object_colors.items():
        print(f"    {obj_name}: BGR{color}")

    # Methods (exclude mask)
    methods = sorted([m for m in methods_set if m.lower() != 'mask'])
    print(f"  Found {len(methods)} methods: {methods}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each method - combine all objects (and all instances) in one image
    for method_name in methods:
        print(f"  Processing method: {method_name}")
        result_img = query_img.copy()

        # Apply each object's instances for this method to the combined image
        for obj_name in objects:
            instance_files = file_map.get(obj_name, {}).get(method_name, [])
            if not instance_files:
                continue

            # Get color for this object
            color = object_colors[obj_name]

            for fname in sorted(instance_files):
                method_img_path = work_dir / fname
                # Load with alpha channel
                method_img = cv2.imread(str(method_img_path), cv2.IMREAD_UNCHANGED)
                if method_img is None:
                    print(f"    Warning: Could not load {fname}")
                    continue

                # Paste foreground with colored border (accumulate on result_img)
                result_img = paste_foreground_with_border(result_img, method_img, color, border_thickness)

        # Save combined result
        output_path = output_dir / f"{method_name}_combined.png"
        cv2.imwrite(str(output_path), result_img)
        print(f"    Saved: {output_path}")
    
    # Process mask separately - combine all object masks with their object-specific colors
    print(f"  Processing mask images")
    
    # Darken the base image once
    result_img = (query_img.copy().astype(np.float32) * 0.5).astype(np.uint8)
    
    mask_found = False
    for obj_name in objects:
        mask_files = file_map.get(obj_name, {}).get('mask', [])
        if not mask_files:
            continue

        # Get color for this object (same as border color)
        color = object_colors[obj_name]

        for fname in sorted(mask_files):
            mask_img_path = work_dir / fname
            # Load with alpha channel
            mask_img = cv2.imread(str(mask_img_path), cv2.IMREAD_UNCHANGED)
            if mask_img is None:
                print(f"    Warning: Could not load {fname}")
                continue

            # Extract foreground mask
            mask = extract_foreground_mask(mask_img)
            mask_bool = mask > 0

            # Set foreground to pure color directly on the darkened image
            for i in range(3):  # BGR channels
                result_img[:, :, i][mask_bool] = color[i]

            mask_found = True
    
    # Save mask result
    if mask_found:
        output_path = output_dir / "mask_combined.png"
        cv2.imwrite(str(output_path), result_img)
        print(f"    Saved: {output_path}")
    
    return True

def find_subdirectories_with_query(root_dir):
    """Find all subdirectories containing query.png"""
    subdirs = []
    root_path = Path(root_dir)
    
    # Walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(root_path):
        if 'query.png' in filenames:
            subdirs.append(Path(dirpath))
    
    return subdirs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize object pose estimation results')
    parser.add_argument('--input-dir', type=str, default='./comparison_results',
                        help='Input directory containing query.png and the per-method overlays produced by '
                             'compare_methods_visualize.py (or a parent directory with multiple such subfolders)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for combined results (default: input_dir/combined_results for each subfolder)')
    parser.add_argument('--border-thickness', type=int, default=3,
                        help='Border thickness for object contours (default: 3)')
    parser.add_argument('--recursive', action='store_false',
                        help='Recursion is ON by default (every subfolder containing query.png is processed); '
                             'pass --recursive to process only the given --input-dir itself')
    
    args = parser.parse_args()
    
    # Set the working directory
    if args.input_dir:
        work_dir = Path(args.input_dir)
    else:
        work_dir = Path(__file__).parent
    
    if not work_dir.exists():
        print(f"Error: Input directory does not exist: {work_dir}")
        return
    
    print(f"Input directory: {work_dir}")
    
    # Determine if we should process recursively or just the single directory
    if args.recursive:
        # Find all subdirectories with query.png
        subdirs = find_subdirectories_with_query(work_dir)
        
        if not subdirs:
            print(f"Error: No subdirectories with query.png found in {work_dir}")
            return
        
        print(f"\nFound {len(subdirs)} subdirectories with query.png:")
        for subdir in subdirs:
            print(f"  - {subdir.relative_to(work_dir)}")
        
        # Process each subdirectory
        total_processed = 0
        total_failed = 0
        
        for subdir in subdirs:
            print(f"\n{'='*80}")
            print(f"Processing: {subdir}")
            print(f"{'='*80}")
            
            # Set output directory for this subdirectory
            if args.output_dir:
                # Use the same relative structure in the output directory
                rel_path = subdir.relative_to(work_dir)
                output_dir = Path(args.output_dir) / rel_path / "combined_results"
            else:
                output_dir = subdir / "combined_results"
            
            print(f"Output directory: {output_dir}")
            
            # Process this directory
            success = process_directory(subdir, output_dir, args.border_thickness)
            
            if success:
                total_processed += 1
                print(f"[OK] Successfully processed: {subdir}")
            else:
                total_failed += 1
                print(f"[FAIL] Failed to process: {subdir}")
        
        print(f"\n{'='*80}")
        print(f"Summary: {total_processed} directories processed successfully, {total_failed} failed")
        print(f"{'='*80}")
    
    else:
        # Process single directory (original behavior)
        # Check if query.png exists in the directory
        query_path = work_dir / "query.png"
        if not query_path.exists():
            print(f"Error: query.png not found in {work_dir}")
            print(f"Tip: Use --recursive flag to process all subdirectories containing query.png")
            return
        
        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = work_dir / "combined_results"
        
        print(f"Output directory: {output_dir}")
        
        # Process the directory
        success = process_directory(work_dir, output_dir, args.border_thickness)
        
        if success:
            print("\n[OK] Processing complete!")
        else:
            print("\n[FAIL] Processing failed!")

if __name__ == "__main__":
    main()
