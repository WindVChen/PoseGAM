#!/bin/bash
# Parallel example (SLURM) for the CONVERT part of step1-download-convert2simpleGLB:
# convert the downloaded raw 3D assets into simple textured GLB meshes with convert.py.
# Each SLURM task takes one GPU and one data shard (rank); together the tasks cover the
# whole dataset (world_size = #tasks). Output meshes are written to
# <OUTPUT_DIR>/converted_meshes/<sha256>/mesh.glb.
#
# This example uses the Toys4k subset. For ObjaverseXL you must additionally pass
# `--source sketchfab` (or `--source github`), see TRELLIS DATASET.md:
# https://github.com/microsoft/TRELLIS/blob/68820295a6ff17b44117d7439d0a244bd9c7826e/DATASET.md

#SBATCH --time=144:00:00
#SBATCH --job-name=convert2glb
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
CONDA_ENV="pytorch"
WORLD_SIZE=7                                        # must match #SBATCH tasks/gpus
# -----------------------------------------

mkdir -p "$WORK_DIR/logs"
export SLURM_JOB_ID=$SLURM_JOB_ID

srun --ntasks=$WORLD_SIZE --gpus-per-task=1 --cpus-per-task=8 bash -c "
    cd \"$WORK_DIR\" || exit 1
    source activate $CONDA_ENV
    export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
    python convert.py $DATASET \
        --output_dir $OUTPUT_DIR \
        --world_size $WORLD_SIZE --rank \$SLURM_PROCID \
        2>&1 | tee \"$WORK_DIR/logs/convert_rank\${SLURM_PROCID}_job\${SLURM_JOB_ID}.log\"
"
