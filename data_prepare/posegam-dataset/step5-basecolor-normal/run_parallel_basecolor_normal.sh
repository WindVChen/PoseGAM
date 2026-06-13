#!/bin/bash
# Parallel example for step5-basecolor-normal: render base-color and normal maps with
# nvdiffrast, reusing the camera poses in each renders3*/<sha256>/transforms.json and the
# watertight meshes. RANK_SIZE processes are launched per mode, each handling one shard.
#
# Two outputs are produced (run both, or comment one out):
#   shading_mode=texture -> <DATA_ROOT>/renders3-color/    (base color, used by the loader)
#   shading_mode=normal  -> <DATA_ROOT>/renders3-normal/   (world-space normals)
source ~/.bashrc  # Ensure conda command is available

# Ensure Ctrl+C kills all subprocesses
trap "echo 'Stopping all processes...'; kill 0" SIGINT

# Navigate to this step's directory (so the relative script path resolves)
cd "$(dirname "$0")" || exit

# Activate conda environment
conda activate pytorch

# ---- Configuration ----
RANK_SIZE=8                                            # number of parallel processes per mode
DATA_ROOT="/ibex/tmp/TRELLIS-500K/Toys4k"
TRANSFORMS_ROOT="$DATA_ROOT/renders3/"                 # camera poses to reuse
MESH_ROOT="$DATA_ROOT/watertight_meshes/"
RESOLUTION=512
DEVICE="cuda"
SCRIPT="nvdiffrast_renderer.py"
# ------------------------

# Render one (shading_mode, output_root) configuration across RANK_SIZE processes.
render_mode () {
    local mode="$1"
    local output_root="$2"
    echo "=== Rendering '$mode' -> $output_root ==="
    for ((RANK_ID=0; RANK_ID<RANK_SIZE; RANK_ID++)); do
        echo "Starting $mode process $RANK_ID out of $RANK_SIZE"
        python "$SCRIPT" \
            --transforms_root "$TRANSFORMS_ROOT" \
            --mesh_root "$MESH_ROOT" \
            --output_root "$output_root" \
            --resolution "$RESOLUTION" \
            --shading_mode "$mode" \
            --device "$DEVICE" \
            --rank_size "$RANK_SIZE" \
            --rank_id "$RANK_ID" \
            > "log_${mode}_rank_$RANK_ID.txt" 2>&1 &
    done
    wait
    echo "=== Finished '$mode' ==="
}

render_mode texture "$DATA_ROOT/renders3-color/"
render_mode normal  "$DATA_ROOT/renders3-normal/"

echo "All rendering processes finished."
