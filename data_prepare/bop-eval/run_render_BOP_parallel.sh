#!/bin/bash
# Parallel example (SLURM) for the BOP reference-view rendering (Blender pass): renders
# multi-view images + depth + transforms.json for every test object, using the watertight
# meshes (obj_<id>/mesh.glb) and the per-case intrinsics from the gigapose image_wise split.
# Run AFTER to_watertight_mesh_BOP.py, and BEFORE the nvdiffrast color/normal pass.

#SBATCH --time=4:00:00
#SBATCH --job-name=render_BOP
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --constraint=v100
#SBATCH --mem=64G
#SBATCH --output=logs/job_%j_main.log
#SBATCH --error=logs/job_%j.err

# ---- Edit these for your environment ----
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"          # this step's directory
DATASET="ycbv"                                     # lmo | tless | tudl | icbin | ycbv
GIGAPOSE_DATASETS="/path/to/gigapose/gigaPose_datasets/datasets"
BOP_DIR="/ibex/tmp/TRELLIS-500K/BOP-data"          # rendered reference views go to $BOP_DIR/$DATASET
LIGHT_MODE="standard"                              # standard | bright
CONDA_ENV="pytorch"
WORLD_SIZE=4                                        # must match #SBATCH tasks/gpus
# -----------------------------------------

# Per-dataset split + FOV setting (matches the settings used for evaluation):
#   tless         : primesense split, fixed 40deg FOV (512x512)
#   ycbv          : test split,       fixed 40deg FOV (512x512)
#   lmo/tudl/icbin: test split,       FOV loaded from the BOP camera intrinsics
case "$DATASET" in
    tless)          SPLIT="test_primesense"; FOV_MODE="fixed40" ;;
    ycbv)           SPLIT="test";            FOV_MODE="fixed40" ;;
    lmo|tudl|icbin) SPLIT="test";            FOV_MODE="loaded"  ;;
    *)              SPLIT="test";            FOV_MODE="fixed40" ;;
esac

mkdir -p "$WORK_DIR/logs"
export SLURM_JOB_ID=$SLURM_JOB_ID

srun --ntasks=$WORLD_SIZE --gpus-per-task=1 --cpus-per-task=8 bash -c "
    cd \"$WORK_DIR\" || exit 1
    source activate $CONDA_ENV
    export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID

    # Virtual display for headless Blender rendering
    export DISPLAY=:\$((99 + \$SLURM_PROCID))
    Xvfb \$DISPLAY -screen 0 1024x768x24 &
    XVFB_PID=\$!
    sleep 2

    python render_BOP.py \
        --input_dir $GIGAPOSE_DATASETS/tmp/${DATASET}_image_wise/$SPLIT \
        --input_glb_dir $BOP_DIR/$DATASET/ \
        --output_dir $BOP_DIR/$DATASET/ \
        --fov_mode $FOV_MODE \
        --light_mode $LIGHT_MODE \
        --num_views 50 \
        --world_size $WORLD_SIZE --rank \$SLURM_PROCID \
        2>&1 | tee \"$WORK_DIR/logs/render_${DATASET}_rank\${SLURM_PROCID}_job\${SLURM_JOB_ID}.log\"

    kill \$XVFB_PID 2>/dev/null || true
"
