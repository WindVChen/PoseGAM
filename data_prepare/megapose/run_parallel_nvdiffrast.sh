#!/bin/bash
# Parallel example for MegaPose data prep: render base-color and normal maps for the
# MegaPose 3D models with nvdiffrast. Camera viewpoints are sampled on a sphere, so no
# transforms.json input is needed; the 'texture' (color) pass writes transforms.json per
# object and the 'normal' pass reuses those poses -- hence color MUST run before normal.
#
# Configure ONE dataset at a time (GSO or ShapeNet) by editing MESH_ROOT / OUTPUT_BASE /
# CAMERA_RADIUS below:
#   GSO      : MESH_ROOT=.../google_scanned_objects/models_normalized/   CAMERA_RADIUS=0.4
#   ShapeNet : MESH_ROOT=.../shapenetcorev2/models_orig/                 CAMERA_RADIUS=0.2
# (the ShapeNet code path is auto-selected when MESH_ROOT contains 'shapenetcorev2').
source ~/.bashrc  # Ensure conda command is available

# Ensure Ctrl+C kills all subprocesses
trap "echo 'Stopping all processes...'; kill 0" SIGINT

# Navigate to this step's directory (so the relative script path resolves)
cd "$(dirname "$0")" || exit

# Activate conda environment
conda activate pytorch

# ---- Configuration (edit per dataset) ----
RANK_SIZE=32                                           # number of parallel processes per mode
MESH_ROOT="/ibex/tmp/TRELLIS-500K/megapose_data/shapenetcorev2/models_orig/"
OUTPUT_BASE="/ibex/tmp/TRELLIS-500K/megapose_data/shapenetcorev2"
CAMERA_RADIUS=0.2                                      # 0.4 for GSO, 0.2 for ShapeNet
RESOLUTION=512
DEVICE="cuda"
SCRIPT="nvdiffrast_renderer.py"
# -------------------------------------------

# Render one (shading_mode, output_root, [extra args]) configuration across RANK_SIZE processes.
render_mode () {
    local mode="$1"
    local output_root="$2"
    local extra="$3"
    echo "=== Rendering '$mode' -> $output_root ==="
    for ((RANK_ID=0; RANK_ID<RANK_SIZE; RANK_ID++)); do
        echo "Starting $mode process $RANK_ID out of $RANK_SIZE"
        python "$SCRIPT" \
            --mesh_root "$MESH_ROOT" \
            --output_root "$output_root" \
            --resolution "$RESOLUTION" \
            --shading_mode "$mode" \
            --device "$DEVICE" \
            --camera_radius "$CAMERA_RADIUS" \
            --rank_size "$RANK_SIZE" \
            --rank_id "$RANK_ID" \
            $extra \
            > "log_${mode}_rank_$RANK_ID.txt" 2>&1 &
    done
    wait
    echo "=== Finished '$mode' ==="
}

# Color first (writes transforms.json + depth maps), then normal (reuses those poses, and
# inherits depth from renders3-color -- so --save_depth is passed only to the color pass).
render_mode texture "$OUTPUT_BASE/renders3-color/" "--save_depth"
render_mode normal  "$OUTPUT_BASE/renders3-normal/"

echo "All rendering processes finished."
