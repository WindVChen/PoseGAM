#!/bin/bash
# Parallel example (SLURM) for step3-renderImages: render multi-view images from the
# watertight meshes with render.py. Each SLURM task takes one GPU and one data shard
# (rank); together the tasks cover the whole dataset (world_size = #tasks).
#
# RENDER_MODE selects the rendering scenario AND the output subfolder:
#   pure     -> <OUTPUT_DIR>/renders3/            (object-centric views, manual lighting)
#   camTrans -> <OUTPUT_DIR>/renders3-camTrans/   (+ random camera translation)
#   envMap   -> <OUTPUT_DIR>/renders3-envMap/     (+ HDR environment-map lighting)
# Run this script once per mode you need.
#
# This example uses the Toys4k subset. For ObjaverseXL additionally pass
# `--source sketchfab` (or `--source github`).

#SBATCH --time=144:00:00
#SBATCH --job-name=renderImages
#SBATCH --gpus=7
#SBATCH --cpus-per-task=8
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=1   # one task (one GPU) per node
#SBATCH --constraint=v100
#SBATCH --mem=128G

# ---- Edit these for your environment ----
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"          # this step's directory
DATASET="Toys4k"                                   # dataset module under datasets/
OUTPUT_DIR="/ibex/tmp/TRELLIS-500K/Toys4k/"
RENDER_MODE="envMap"                               # pure | camTrans | envMap
CONDA_ENV="pytorch"
WORLD_SIZE=7                                        # must match #SBATCH tasks/gpus
# -----------------------------------------

mkdir -p "$WORK_DIR/logs"
export SLURM_JOB_ID=$SLURM_JOB_ID

srun --ntasks=$WORLD_SIZE --gpus-per-task=1 --cpus-per-task=8 bash -c "
    cd \"$WORK_DIR\" || exit 1
    source activate $CONDA_ENV
    export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
    python render.py $DATASET \
        --output_dir $OUTPUT_DIR \
        --render_mode $RENDER_MODE \
        --world_size $WORLD_SIZE --rank \$SLURM_PROCID \
        2>&1 | tee \"$WORK_DIR/logs/render_${RENDER_MODE}_rank\${SLURM_PROCID}_job\${SLURM_JOB_ID}.log\"
"
