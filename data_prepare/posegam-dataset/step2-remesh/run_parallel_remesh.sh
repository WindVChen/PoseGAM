#!/bin/bash
source ~/.bashrc  # Ensure conda command is available

# Ensure Ctrl+C kills all subprocesses
trap "echo 'Stopping all processes...'; kill 0" SIGINT

# Navigate to this step's directory (so the relative script path resolves)
cd "$(dirname "$0")" || exit

# Activate conda environment
conda activate pytorch

# Configuration
RANK_SIZE=8  # Change this to however many processes you want to run
RESOLUTION=128
DATA_ROOT="/ibex/tmp/TRELLIS-500K/Toys4k"
JSON_FILE="$DATA_ROOT/converted_meshes/mesh_path.json"   # produced by detect_path.py
REMESH_DIR="$DATA_ROOT/watertight_meshes"

# Path to the Python script
SCRIPT="to_watertight_mesh.py"

for ((RANK_ID=0; RANK_ID<RANK_SIZE; RANK_ID++))
do
  echo "Starting process $RANK_ID out of $RANK_SIZE"
  python "$SCRIPT" \
    --resolution "$RESOLUTION" \
    --json_file_path "$JSON_FILE" \
    --remesh_target_path "$REMESH_DIR" \
    --rank_size "$RANK_SIZE" \
    --rank_id "$RANK_ID" \
    > "log_rank_$RANK_ID.txt" 2>&1 &  # log stdout/stderr per rank
done

wait  # Wait for all background processes to complete
echo "All processes finished."
