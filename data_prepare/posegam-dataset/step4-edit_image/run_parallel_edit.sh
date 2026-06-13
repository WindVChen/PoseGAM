#!/bin/bash

#SBATCH --time=144:00:00
#SBATCH --job-name=edit-image-flux-64ranks
#SBATCH --gpus=8
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1   # 1 tasks per node, each using 1 GPU
#SBATCH --cpus-per-task=4
#SBATCH --constraint=a100
#SBATCH --mem=32G
#SBATCH --output=logs/job_%j_edit_image.log
#SBATCH --error=logs/job_%j_edit_image.err

# Image editing using FLUX.1-Canny-dev model with parallel processing
# Processes images from ObjaverseXL_Toys4k dataset with pitch filtering and visibility checking
#
# Features:
# - Pitch filtering: 15-60° above object (positive elevation)
# - Yaw filtering: Front-facing cameras only (configurable angular range)
# - Background masking: Preserves original RGBA alpha channel
# - Debug mode: Saves original, Canny edge detection, and final edited images
# - Parallel processing: Multi-GPU support with rank-based distribution
#
# Configurable parameters:
# - PITCH_MIN/PITCH_MAX: Camera elevation angle range
# - FRONT_ANGLE_RANGE: Front-facing angular range (±60° from front = 120°)
# - NUM_INFERENCE_STEPS: FLUX inference steps (higher = better quality, slower)
# - GUIDANCE_SCALE: FLUX guidance scale (higher = stronger prompt adherence)
# - ENABLE_DEBUG: Save debug images (original, Canny, final)

# !!Note: 3000 samples processing requires ~5 hour using this script for normal images. ~1.5 hour for camTrans images.

# Configuration
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"          # this step's directory
mkdir -p "$WORK_DIR/logs"

TOTAL_RANKS=16
RANKS_PER_GPU=2
# Edit one input/output pair per run: renders3 -> renders3-edited, or
# renders3-camTrans -> renders3-camTrans-edited.
INPUT_DIR="/ibex/tmp/TRELLIS-500K/Toys4k/renders3"
OUTPUT_DIR="/ibex/tmp/TRELLIS-500K/Toys4k/renders3-edited"
METADATA_CSV="/ibex/tmp/TRELLIS-500K/Toys4k/metadata.csv"
SCRIPT="$WORK_DIR/edit_image_with_edge.py"

# FLUX parameters
PITCH_MIN=15.0
PITCH_MAX=60.0
FRONT_ANGLE_RANGE=120.0
NUM_INFERENCE_STEPS=28
GUIDANCE_SCALE=30.0
ENABLE_DEBUG=true  # Set to true to enable debug mode (saves original and Canny images)

echo "Starting image editing job with FLUX.1-Canny-dev"
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Metadata file: $METADATA_CSV"
echo "Total ranks: $TOTAL_RANKS"
echo "Ranks per GPU: $RANKS_PER_GPU"
echo "Pitch range: ${PITCH_MIN}° to ${PITCH_MAX}°"
echo "Front angle range: ${FRONT_ANGLE_RANGE}°"
echo "FLUX parameters: steps=$NUM_INFERENCE_STEPS, guidance=$GUIDANCE_SCALE"
echo "Debug mode: $ENABLE_DEBUG"

# Run 4 tasks (2 per node across 2 nodes), each handling 2 ranks
srun --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 bash -c "
    cd \"$WORK_DIR\" || exit 1
    source ~/.bashrc
    conda activate pytorch
    
    # Calculate rank range for this GPU
    GPU_ID=\$SLURM_PROCID
    START_RANK=\$((GPU_ID * $RANKS_PER_GPU))
    END_RANK=\$((START_RANK + $RANKS_PER_GPU))
    
    # SLURM with --gpus-per-task=1 automatically assigns one GPU per task
    # Each task sees only GPU 0, so we don't need to set CUDA_VISIBLE_DEVICES
    
    echo \"Task \$GPU_ID (SLURM_PROCID \$SLURM_PROCID) handling ranks \$START_RANK to \$((END_RANK-1))\"
    echo \"Working directory: \$(pwd)\"
    echo \"Script path: $SCRIPT\"
    echo \"Node: \$SLURMD_NODENAME\"
    
    # Check if required files exist
    if [ ! -f \"$SCRIPT\" ]; then
        echo \"Error: Script not found at $SCRIPT\"
        exit 1
    fi
    
    if [ ! -d \"$INPUT_DIR\" ]; then
        echo \"Error: Input directory not found at $INPUT_DIR\"
        exit 1
    fi
    
    if [ ! -f \"$METADATA_CSV\" ]; then
        echo \"Error: Metadata file not found at $METADATA_CSV\"
        exit 1
    fi
    
    # Start 2 ranks in parallel for this GPU
    for ((RANK_ID=START_RANK; RANK_ID<END_RANK; RANK_ID++))
    do
        echo \"Starting rank \$RANK_ID on task \$GPU_ID (node \$SLURMD_NODENAME)\"
        
        # Build debug argument (note: --debug flag DISABLES debug mode due to store_false action)
        DEBUG_ARG=\"\"
        if [ \"$ENABLE_DEBUG\" = \"true\" ]; then
            # Don't add --debug flag, so debug mode stays enabled (default is True)
            DEBUG_ARG=\"\"
        else
            # Add --debug flag to disable debug mode
            DEBUG_ARG=\"--debug\"
        fi
        
        python \"$SCRIPT\" \\
            --input_dir \"$INPUT_DIR\" \\
            --output_dir \"$OUTPUT_DIR\" \\
            --metadata_csv \"$METADATA_CSV\" \\
            --rank_size \"$TOTAL_RANKS\" \\
            --rank_id \"\$RANK_ID\" \\
            --pitch_min \"$PITCH_MIN\" \\
            --pitch_max \"$PITCH_MAX\" \\
            --front_angle_range \"$FRONT_ANGLE_RANGE\" \\
            --num_inference_steps \"$NUM_INFERENCE_STEPS\" \\
            --guidance_scale \"$GUIDANCE_SCALE\" \\
            \$DEBUG_ARG \\
            > \"$WORK_DIR/logs/rank_\${RANK_ID}_gpu_\${GPU_ID}_job_\${SLURM_JOB_ID}_edit.log\" 2>&1 &
    done
    
    # Wait for all ranks on this GPU to complete
    wait
    echo \"All ranks (\$START_RANK to \$((END_RANK-1))) on task \$GPU_ID completed\"
"

echo "All tasks and ranks completed successfully"
echo "Check individual rank logs in: $WORK_DIR/logs/"
