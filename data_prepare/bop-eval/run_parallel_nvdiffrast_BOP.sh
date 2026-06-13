#!/bin/bash
# Parallel example for the BOP base-color + normal rendering (nvdiffrast pass). It reuses
# the per-object transforms.json written by render_BOP.py, so run that Blender pass FIRST.
# Both passes read the same poses, so order between color/normal does not matter here.
source ~/.bashrc  # Ensure conda command is available

# Ensure Ctrl+C kills all subprocesses
trap "echo 'Stopping all processes...'; kill 0" SIGINT

# Navigate to this step's directory (so the relative script path resolves)
cd "$(dirname "$0")" || exit

# Activate conda environment
conda activate pytorch

# ---- Configuration ----
RANK_SIZE=8                                        # number of parallel processes per mode
DATASET="ycbv"                                     # lmo | tless | tudl | icbin | ycbv
BOP_DIR="/ibex/tmp/TRELLIS-500K/BOP-data"
TRANSFORMS_ROOT="$BOP_DIR/$DATASET/"               # reference views + transforms.json
MESH_ROOT="$BOP_DIR/$DATASET/"                     # obj_<id>/mesh.glb live here too
RESOLUTION=512
DEVICE="cuda"
SCRIPT="nvdiffrast_renderer_BOP.py"
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
            > "log_${DATASET}_${mode}_rank_$RANK_ID.txt" 2>&1 &
    done
    wait
    echo "=== Finished '$mode' ==="
}

render_mode texture "$BOP_DIR/$DATASET-color/"
render_mode normal  "$BOP_DIR/$DATASET-normal/"

echo "All rendering processes finished."
