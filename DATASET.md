# Dataset Preparation

This document covers **all** data used by PoseGAM:

1. **Main training dataset** — the multi-view dataset rendered from raw
   [TRELLIS-500K](https://github.com/microsoft/TRELLIS/blob/68820295a6ff17b44117d7439d0a244bd9c7826e/DATASET.md)
   3D assets (the five-stage pipeline below).
2. **Optional MegaPose training data** (Google Scanned Objects + ShapeNet) —
   see [Optional: training on existing MegaPose data](#optional-training-on-existing-megapose-data).
3. **BOP evaluation data** (LMO, T-LESS, TUDL, IC-BIN, YCB-V) —
   see [Optional: BOP evaluation data](#optional-bop-evaluation-data).

---

## Main training dataset

The main pipeline turns raw TRELLIS-500K 3D assets into the multi-view rendered dataset
consumed by the training code (see `posegam/training/data/datasets/posegam_dataset.py`).

It has five stages. Each stage lives in its own folder and writes into a single
**dataset root** (e.g. `/ibex/tmp/TRELLIS-500K/Toys4k/`):

```
                                                   ┌─ renders3/            (pure)
 raw assets ─► converted_meshes/ ─► watertight_    │─ renders3-camTrans/   (+ cam translation)
 (download)     (convert.py)        meshes/  ──────┤─ renders3-envMap/     (+ HDR env map)
                                    (remesh)        │
                                                    ├─ renders3-edited/          (FLUX edit of renders3)
                                                    ├─ renders3-camTrans-edited/ (FLUX edit of renders3-camTrans)
                                                    │
                                                    ├─ renders3-color/     (base color, nvdiffrast)
                                                    └─ renders3-normal/     (normals, nvdiffrast)
```

| Step | Folder | Entry script | Produces |
|------|--------|--------------|----------|
| 1. Download | `data_prepare/posegam-dataset/step1-download-convert2simpleGLB/` | `download.py`, `build_metadata.py` | `raw/`, `metadata.csv` |
| 2. Convert  | `data_prepare/posegam-dataset/step1-download-convert2simpleGLB/` | `convert.py` | `converted_meshes/` |
| 3. Remesh   | `data_prepare/posegam-dataset/step2-remesh/` | `detect_path.py`, `to_watertight_mesh.py` | `watertight_meshes/` |
| 4. Render   | `data_prepare/posegam-dataset/step3-renderImages/` | `render.py` | `renders3/`, `renders3-camTrans/`, `renders3-envMap/` |
| 5. Edit     | `data_prepare/posegam-dataset/step4-edit_image/` | `edit_image_with_edge.py` | `renders3-edited/`, `renders3-camTrans-edited/` |
| 6. Color/Normal | `data_prepare/posegam-dataset/step5-basecolor-normal/` | `nvdiffrast_renderer.py` | `renders3-color/`, `renders3-normal/` |

All examples below use the **`Toys4k`** subset and a dataset root `DATA=/path/to/Toys4k`.
For the **`ObjaverseXL`** subset, pass an extra `--source sketchfab` (or `--source github`) to
the dataset scripts, as described in the [TRELLIS DATASET.md](https://github.com/microsoft/TRELLIS/blob/68820295a6ff17b44117d7439d0a244bd9c7826e/DATASET.md).

Every step ships a parallel-run shell script (SLURM or plain multi-process) — see the
"parallel" note under each step. They shard work with `--rank` / `--world_size` (or
`--rank_id` / `--rank_size`) so the dataset can be split across GPUs/nodes.

---

## Environment

All steps run in the **single project environment** from the
[Installation](README.md#installation) section — the data-preparation-only packages
(`bpy`, `point_cloud_utils`, `diffusers`, `controlnet_aux`, `objaverse`, `utils3d`, plus the
CUDA-compiled `cubvh` + `diso`) are listed in [`requirements.txt`](requirements.txt).

Two extra, non-pip requirements:
- **Blender 4.2** (rendering, steps 2 & 4): the scripts auto-download it on first run, or set
  the `BLENDER_PATH` environment variable to point at your own install.
- **FLUX.1-Canny-dev** (step 5, image editing): weights download automatically on first use;
  needs a GPU with ≈12 GB+ memory.

---

## Step 1 — Download raw TRELLIS-500K data

Follows steps 1–3 of the TRELLIS repo's [Guidance](https://github.com/microsoft/TRELLIS/blob/68820295a6ff17b44117d7439d0a244bd9c7826e/DATASET.md) to download the raw 3D assets. The download scripts are also copied to `data_prepare/posegam-dataset/step1-download-convert2simpleGLB/`:

```bash
DATA=/path/to/Toys4k

# 1) build the initial metadata table
python build_metadata.py Toys4k --output_dir $DATA

# 2) download the raw assets (shardable with --rank / --world_size)
python download.py Toys4k --output_dir $DATA --world_size 1 --rank 0

# 3) refresh metadata so `local_path` points at the downloaded files
python build_metadata.py Toys4k --output_dir $DATA
```

> `Toys4k` must be downloaded manually — `download.py` prints the instructions and the URL
> to fetch `toys4k_blend_files.zip` into `$DATA/raw/`.

## Step 2 — Convert raw assets to simple textured GLB

`convert.py` loads each asset in Blender, merges/triangulates it, bakes a single texture and
exports `mesh.glb`. From `data_prepare/posegam-dataset/step1-download-convert2simpleGLB/`:

```bash
python convert.py Toys4k --output_dir $DATA --world_size 1 --rank 0
# -> $DATA/converted_meshes/<sha256>/mesh.glb
```

**Parallel:** `run_parallel_convert.sh` (SLURM, one GPU per rank).

## Step 3 — Remesh into watertight meshes

First collect the converted meshes into a path list, then remesh them to watertight,
normalized meshes. From `data_prepare/posegam-dataset/step2-remesh/`:

```bash
# 1) list every converted_meshes/<sha256>/mesh.glb
python detect_path.py \
    --directory_to_search $DATA/converted_meshes \
    --json_file_path      $DATA/converted_meshes/mesh_path.json \
    --file_type .glb

# 2) remesh (--resolution is the UDF grid resolution; default 128, as used in our scripts)
python to_watertight_mesh.py \
    --resolution 128 \
    --json_file_path     $DATA/converted_meshes/mesh_path.json \
    --remesh_target_path $DATA/watertight_meshes
# -> $DATA/watertight_meshes/<sha256>/mesh.glb
```

**Parallel:** `run_parallel_remesh.sh` (multi-process, shards by `--rank_id` / `--rank_size`;
also uses `RESOLUTION=128`).

## Step 4 — Render multi-view images

`render.py` renders the watertight meshes from Hammersley-sampled viewpoints. The
`--render_mode` switch selects both the rendering scenario and the output subfolder.
Run it once per mode you need. From `data_prepare/posegam-dataset/step3-renderImages/`:

```bash
python render.py Toys4k --output_dir $DATA --render_mode pure      # -> renders3/
python render.py Toys4k --output_dir $DATA --render_mode camTrans  # -> renders3-camTrans/
python render.py Toys4k --output_dir $DATA --render_mode envMap    # -> renders3-envMap/
```

| `--render_mode` | Adds | Output |
|-----------------|------|--------|
| `pure` | object-centric views, manual lighting | `renders3/` |
| `camTrans` | + random camera translation | `renders3-camTrans/` |
| `envMap` | + HDR environment-map lighting | `renders3-envMap/` |

Each `<sha256>/` folder gets `NNN.png`, `NNN_depth.png` and a `transforms.json`.

> **HDR maps for `envMap` mode.** The `envMap` mode lights each object with a random HDRI.
> Download a set of HDR environment maps (we use the CC0 1K `.hdr` maps from
> [Poly Haven](https://polyhaven.com/hdris)) into a folder, e.g. `collected_hdr_files/`, and
> point the renderer at it via `--environment_map_folder /path/to/collected_hdr_files`
> (the blender script picks one `.hdr` per view at random). You can fetch them with the
> [Poly Haven API](https://github.com/Poly-Haven/Public-API) or `pip install pyhaven`; a few
> hundred maps are plenty.

**Parallel:** `run_parallel_render.sh` (SLURM; set `RENDER_MODE` at the top).

## Step 5 — Edit rendered images (FLUX.1-Canny-dev)

`edit_image_with_edge.py` re-textures the front-facing, slightly-elevated views with
edge-controlled FLUX generation, keeping geometry and the transparent background. Run it
on the `renders3` and `renders3-camTrans` sets. From `data_prepare/posegam-dataset/step4-edit_image/`:

```bash
python edit_image_with_edge.py \
    --input_dir  $DATA/renders3 \
    --output_dir $DATA/renders3-edited \
    --metadata_csv $DATA/metadata.csv

python edit_image_with_edge.py \
    --input_dir  $DATA/renders3-camTrans \
    --output_dir $DATA/renders3-camTrans-edited \
    --metadata_csv $DATA/metadata.csv
```

Edited folders contain only the kept frames and **no depth maps** — the training loader
reuses depth from the source folder (`renders3-edited` ↔ `renders3`,
`renders3-camTrans-edited` ↔ `renders3-camTrans`).

**Parallel:** `run_parallel_edit.sh` (SLURM, shards by `--rank_id` / `--rank_size`).

## Step 6 — Render base-color and normal maps

`nvdiffrast_renderer.py` re-renders the `renders3` camera poses against the watertight
meshes to produce per-pixel base color and normals. Run it **twice** — once per shading
mode. From `data_prepare/posegam-dataset/step5-basecolor-normal/`:

```bash
# base color (shader mode 'texture')
python nvdiffrast_renderer.py \
    --transforms_root $DATA/renders3/ \
    --mesh_root       $DATA/watertight_meshes/ \
    --output_root     $DATA/renders3-color/ \
    --shading_mode texture

# normals
python nvdiffrast_renderer.py \
    --transforms_root $DATA/renders3/ \
    --mesh_root       $DATA/watertight_meshes/ \
    --output_root     $DATA/renders3-normal/ \
    --shading_mode normal
```

**Parallel:** `run_parallel_basecolor_normal.sh` renders both modes across `RANK_SIZE` processes.

---

## Using the prepared data for training

Point the dataset `data_root` at the **`renders3/`** folder of a dataset root. The loader
discovers the sibling folders (`renders3-camTrans`, `renders3-envMap`, the `*-edited`
variants, `renders3-color`, `renders3-normal`) automatically by string-replacing `renders3`,
so the directory names above must be kept as-is.

---

## Optional: training on existing MegaPose data

The training code can additionally train on the [MegaPose](https://github.com/megapose6d/megapose6d)
Google Scanned Objects (GSO) and ShapeNetCoreV2 sets (loaders:
`posegam/training/data/datasets/megapose.py`). Preparation code lives in
`data_prepare/megapose/`. It uses the same project environment (nvdiffrast).

### 1. Download the MegaPose assets

Download into `/ibex/tmp/TRELLIS-500K/megapose_data/`:

- **Real image webdatasets** (provide `extracted/`, `key_to_shard.json` and the `*_models.json`):
  - `gso_1M`  → `megapose_data/GSO/`
    (from <https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/webdatasets/gso_1M/>)
  - `shapenet_1M` → `megapose_data/ShapeNet/`
    (from <https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/webdatasets/shapenet_1M/>)
- **3D model archives** (from <https://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/tars/>), extracted so that:
  - `google_scanned_objects.zip` → `megapose_data/google_scanned_objects/models_normalized/`
  - `shapenetcorev2.zip`         → `megapose_data/shapenetcorev2/models_orig/`

### 2. Render base color + normal maps

`data_prepare/megapose/nvdiffrast_renderer.py` samples camera viewpoints on a sphere and
renders the 3D models. As in step 6 it is run per shading mode, and the `texture` (color)
pass must run before the `normal` pass (the normal pass reuses the poses written by the
color pass). GSO and ShapeNet are auto-detected from `--mesh_root` and differ in camera
radius and mesh layout:

```bash
# Google Scanned Objects (camera_radius 0.4)
python nvdiffrast_renderer.py --shading_mode texture --camera_radius 0.4 \
    --mesh_root   /ibex/tmp/TRELLIS-500K/megapose_data/google_scanned_objects/models_normalized/ \
    --output_root /ibex/tmp/TRELLIS-500K/megapose_data/google_scanned_objects/renders3-color/
python nvdiffrast_renderer.py --shading_mode normal  --camera_radius 0.4 \
    --mesh_root   /ibex/tmp/TRELLIS-500K/megapose_data/google_scanned_objects/models_normalized/ \
    --output_root /ibex/tmp/TRELLIS-500K/megapose_data/google_scanned_objects/renders3-normal/

# ShapeNetCoreV2 (camera_radius 0.2)
python nvdiffrast_renderer.py --shading_mode texture --camera_radius 0.2 \
    --mesh_root   /ibex/tmp/TRELLIS-500K/megapose_data/shapenetcorev2/models_orig/ \
    --output_root /ibex/tmp/TRELLIS-500K/megapose_data/shapenetcorev2/renders3-color/
python nvdiffrast_renderer.py --shading_mode normal  --camera_radius 0.2 \
    --mesh_root   /ibex/tmp/TRELLIS-500K/megapose_data/shapenetcorev2/models_orig/ \
    --output_root /ibex/tmp/TRELLIS-500K/megapose_data/shapenetcorev2/renders3-normal/
```

**Parallel:** `run_parallel_nvdiffrast.sh` runs both the color and normal passes for one
dataset across `RANK_SIZE` processes (edit `MESH_ROOT` / `OUTPUT_BASE` / `CAMERA_RADIUS` for
GSO vs ShapeNet).

### 3. Use it for training

Point the loader's `data_root` at the rendered **`renders3-color/`** folder
(`.../google_scanned_objects/renders3-color` or `.../shapenetcorev2/renders3-color`). The
loader finds the sibling `renders3-normal/` and the real MegaPose images under
`../../GSO/extracted` / `../../ShapeNet/extracted` automatically, so keep the directory
layout above.

---

## Optional: BOP evaluation data

To evaluate on the BOP benchmark (`posegam/evaluation/test_BOP_benchmark.py`) we prepare,
for each object instance in the test images, a set of rendered **reference views** (color +
normal + depth) from the object's CAD model. Preparation code is in `data_prepare/bop-eval/`.
Supported datasets: **LMO, T-LESS, TUDL, IC-BIN, YCB-V**. This uses the same project
environment (Blender + nvdiffrast).

Two paths are used throughout this section:

```bash
# BOP_DIR: working root for this step. All rendered reference views, the per-object meshes,
#          and the CNOS detections live here; the evaluation later reads it via --BOP_dir.
BOP_DIR=/path/to/BOP-data
# GIGA: gigapose datasets directory, populated by gigapose's download in step 1 below.
GIGA=/path/to/gigapose/gigaPose_datasets/datasets
```

The reference views are rendered with **per-dataset settings** (these are the settings used
to produce the evaluation results). The examples below use **YCBV**; for other datasets use
the column for that dataset:

| Dataset | image-wise split | `--fov_mode` | Mesh handling (auto from path) |
|---------|------------------|--------------|--------------------------------|
| ycbv  | `test`            | `fixed40` | keep original UV texture |
| tless | `test_primesense` | `fixed40` | flat gray vertex color |
| lmo / tudl / icbin | `test` | `loaded` | remesh + bake texture |

- `--fov_mode fixed40` renders square 512×512 at a fixed 40° FOV; `--fov_mode loaded` renders
  at the native query-image resolution with the FOV from the BOP camera intrinsics.
- Mesh handling is selected automatically from the dataset name in the model path
  (`tless`→gray, `ycbv`→textured, otherwise remesh+bake) — no flag needed.
- Lighting defaults to `--light_mode standard`; a brighter `--light_mode bright` is also
  available.
- The provided run scripts (`run_render_BOP_parallel.sh`) set the split and `--fov_mode`
  automatically per `DATASET`.

### 1. Download the BOP test data (via gigapose)

Clone and set up [gigapose](https://github.com/nv-nguyen/gigapose), then download the BOP-23
test split. This fetches the test images, **CAD models**, and `test_targets_bop19.json`, and
converts the images to the image-wise layout the prep/eval code reads:

```bash
# inside the gigapose repo
python -m src.scripts.download_test_bop23
# -> $GIGA/<dataset>/models/obj_000001.ply ...        (CAD models + test_targets_bop19.json)
# -> $GIGA/tmp/<dataset>_image_wise/test[_primesense]/ (per-case query images + intrinsics)
```

> Only `download_test_bop23` is needed. gigapose's `download_bop_templates` /
> `render_bop_templates` are **not** used — we render our own reference views below.

### 2. Download the CNOS detections

The eval uses the default BOP-23 CNOS-FastSAM detections. Download and extract the
`cnos-fastsam/` folder into `$BOP_DIR/`:

```bash
wget https://bop.felk.cvut.cz/media/data/bop_datasets_extra/bop23_default_detections_for_task4.zip
unzip bop23_default_detections_for_task4.zip      # provides a cnos-fastsam/ folder
# place it so that: $BOP_DIR/cnos-fastsam/cnos-fastsam_<dataset>-test_*.json
```

### 3. Convert CAD models to watertight GLB

```bash
cd data_prepare/bop-eval
python to_watertight_mesh_BOP.py \
    --obj_dir_path       $GIGA/ycbv/models \
    --remesh_target_path $BOP_DIR/ycbv/
# -> $BOP_DIR/ycbv/obj_<id:06d>/mesh.glb
```

### 4. Render reference views (Blender pass)

For each test (scene, image) this renders the related objects' meshes from sampled
viewpoints, writing the camera poses to `transforms.json`. `--input_dir` points at the
gigapose image-wise split (used for per-case intrinsics). Set `--fov_mode` per the table
above (ycbv → `fixed40`):

```bash
python render_BOP.py \
    --input_dir     $GIGA/tmp/ycbv_image_wise/test \
    --input_glb_dir $BOP_DIR/ycbv/ \
    --output_dir    $BOP_DIR/ycbv/ \
    --fov_mode fixed40
# -> $BOP_DIR/ycbv/<scene>_<image>/<obj_id>/{NNN.png, NNN_depth.png, transforms.json}
```

**Parallel:** `run_render_BOP_parallel.sh` (SLURM; sets up Xvfb for headless Blender and
auto-selects the split / `--fov_mode` from `DATASET`).

### 5. Render base color + normal (nvdiffrast pass)

Reuses the `transforms.json` from step 4, so it must run after it. Run once per mode:

```bash
python nvdiffrast_renderer_BOP.py --shading_mode texture \
    --transforms_root $BOP_DIR/ycbv/ --mesh_root $BOP_DIR/ycbv/ \
    --output_root     $BOP_DIR/ycbv-color/
python nvdiffrast_renderer_BOP.py --shading_mode normal \
    --transforms_root $BOP_DIR/ycbv/ --mesh_root $BOP_DIR/ycbv/ \
    --output_root     $BOP_DIR/ycbv-normal/
```

**Parallel:** `run_parallel_nvdiffrast_BOP.sh` runs both passes for one dataset.

### Resulting layout

```
$BOP_DIR/
├── cnos-fastsam/cnos-fastsam_<dataset>-test_*.json
├── <dataset>/
│   ├── obj_<id:06d>/mesh.glb                                   # watertight CAD (step 3)
│   └── <scene>_<image>/<obj_id>/{NNN.png, NNN_depth.png, transforms.json}   # step 4
├── <dataset>-color/<scene>_<image>/<obj_id>/NNN.png           # step 5 (texture)
└── <dataset>-normal/<scene>_<image>/<obj_id>/NNN.png          # step 5 (normal)
```

Once `$BOP_DIR` is prepared as above, **running the benchmark and the qualitative
visualization is described in the main README** — see its
[Evaluation](README.md#evaluation) and [Visualization](README.md#visualization) sections.

## Acknowledgements

The remeshing step builds on the [Dora](https://github.com/Seed3D/Dora) team's code,
[cubvh](https://github.com/ashawkey/cubvh) (fast UDF) and
[diso](https://github.com/SarahWeiii/diso) (fast iso-surface extraction). The TRELLIS-500K
metadata/download toolkit is from [TRELLIS](https://github.com/microsoft/TRELLIS), and the
optional MegaPose data is from [MegaPose](https://github.com/megapose6d/megapose6d). BOP
evaluation data uses the [BOP toolkit](https://bop.felk.cvut.cz/) and the
[gigapose](https://github.com/nv-nguyen/gigapose) download/conversion scripts.
