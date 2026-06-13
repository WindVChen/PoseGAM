import os
import json
import copy
import sys
import importlib
import argparse
import cv2
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call
import numpy as np
from typing import *
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def get_file_hash(file: str) -> str:
    sha256 = hashlib.sha256()
    # Read the file from the path
    with open(file, "rb") as f:
        # Update the hash with the file content
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256.update(byte_block)
    return sha256.hexdigest()

# ===============LOW DISCREPANCY SEQUENCES================

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]



BLENDER_LINK = 'https://download.blender.org/release/Blender4.2/blender-4.2.1-linux-x64.tar.xz'
BLENDER_INSTALLATION_PATH = '/ibex/tmp/TRELLIS-500K/'
# Path to the Blender executable. Override via the BLENDER_PATH environment variable.
BLENDER_PATH = os.environ.get('BLENDER_PATH', os.path.join(BLENDER_INSTALLATION_PATH, 'blender-4.2.1-linux-x64', 'blender'))

def _install_blender():
    if not os.path.exists(BLENDER_PATH):
        # os.system('sudo apt-get update')
        # os.system('sudo apt-get install -y libxrender1 libxi6 libxkbcommon-x11-0 libsm6')
        os.system(f'wget {BLENDER_LINK} -P {BLENDER_INSTALLATION_PATH}')
        os.system(f'tar -xvf {BLENDER_INSTALLATION_PATH}/blender-4.2.1-linux-x64.tar.xz -C {BLENDER_INSTALLATION_PATH}')


def _render(file_path, num_views, height, width, fovx, fovy, output_folder, fov_mode='fixed40', light_mode='standard'):
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [4] * num_views

    # FOV mode controls both the field of view and the render resolution:
    #   'fixed40' : square 512x512 renders at a fixed 40deg FOV
    #   'loaded'  : native query-image resolution + the (anisotropic) FOV
    #               derived from the BOP camera intrinsics
    if fov_mode == 'loaded':
        fovxs = [fovx] * num_views
        fovys = [fovy] * num_views
    else:  # 'fixed40'
        width = height = 512
        fovxs = fovys = [40 / 180 * np.pi] * num_views

    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fovx': fx, 'fovy': fy} for y, p, r, fx, fy in zip(yaws, pitchs, radius, fovxs, fovys)]

    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'render_blender_BOP.py'),
        '--',
        '--views', json.dumps(views),
        '--object', os.path.expanduser(file_path),
        '--resolutionx', str(width),
        '--resolutiony', str(height),
        '--output_folder', output_folder,
        '--engine', 'BLENDER_EEVEE_NEXT',
        '--light_mode', light_mode,
        '--save_depth',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)
    
    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, 'render.log')
    
    # Set environment variables for headless rendering
    env = os.environ.copy()
    env['LIBGL_ALWAYS_SOFTWARE'] = '1'  # Force software rendering if needed
    env['PYOPENGL_PLATFORM'] = 'egl'     # Use EGL instead of GLX
    
    with open(log_file, 'w') as log:
        call(args, stdout=log, stderr=log, env=env)

import glob
import json

def process_single_case(case, opt, valid_targets):
    """Process a single test case and render all related objects"""
    case_rgb = cv2.imread(os.path.join(opt.input_dir, case + '.rgb.png'))

    height, width = case_rgb.shape[:2]

    camera_info = json.load(open(os.path.join(opt.input_dir, case + '.camera.json'), 'r'))["cam_K"]
    fx, fy = camera_info[0], camera_info[4]

    fovx = 2 * np.arctan(width / (2 * fx))
    fovy = 2 * np.arctan(height / (2 * fy))

    scene_id, im_id = int(case.split('_')[0]), int(case.split('_')[1])

    # Collect related objects from valid_targets with O(1) lookup
    related_obj = valid_targets.get((scene_id, im_id), set())

    # related_obj = set([x['obj_id'] for x in json.load(open(os.path.join(opt.input_dir, case + '.gt.json'), 'r'))])

    # Only process if there are related objects
    if not related_obj:
        return {'case': case, 'status': 'skipped', 'reason': 'no related objects'}
    
    out_case_dir = os.path.join(opt.output_dir, case)
    os.makedirs(out_case_dir, exist_ok=True)

    for obj_id in related_obj:
        out_obj_dir = os.path.join(out_case_dir, str(obj_id))
        os.makedirs(out_obj_dir, exist_ok=True)

        # if os.path.exists(os.path.join(out_obj_dir, 'transforms.json')):
        #     continue

        obj_file = os.path.join(opt.input_glb_dir, f'obj_{obj_id:06d}', 'mesh.glb')

        _render(obj_file, opt.num_views, height, width, fovx, fovy, out_obj_dir,
                fov_mode=opt.fov_mode, light_mode=opt.light_mode)
    
    return {'case': case, 'status': 'success'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/path/to/gigapose/gigaPose_datasets/datasets/tmp/ycbv_image_wise/test',
                        help='gigapose image_wise test split (provides per-case intrinsics: <case>.camera.json)')
    parser.add_argument('--input_glb_dir', type=str, default='/ibex/tmp/TRELLIS-500K/BOP-data/ycbv/',
                        help='Directory with the watertight per-object meshes (obj_<id:06d>/mesh.glb)')
    parser.add_argument('--output_dir', type=str, default='/ibex/tmp/TRELLIS-500K/BOP-data/ycbv/',
                        help='Output directory for rendered reference views (<BOP_dir>/<dataset>/)')
    parser.add_argument('--fov_mode', type=str, default='fixed40', choices=['fixed40', 'loaded'],
                        help="Camera FOV/resolution setting. 'fixed40': square 512x512 renders at a fixed "
                             "40deg FOV (recommended for tless and ycbv); 'loaded': render at the native "
                             "query-image resolution and the (anisotropic) FOV derived from the BOP camera "
                             "intrinsics (recommended for lmo, tudl, icbin).")
    parser.add_argument('--light_mode', type=str, default='standard', choices=['standard', 'bright'],
                        help="Scene lighting. 'standard': key + top/bottom area lights; 'bright': a denser "
                             "point-light grid giving a brighter scene.")
    parser.add_argument('--num_views', type=int, default=50,
                        help='Number of views to render')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total number of parallel processes')
    parser.add_argument('--rank', type=int, default=0,
                        help='Rank of current process')
    parser.add_argument('--max_workers', type=int, default=8,
                        help='Number of parallel workers for rendering')
    opt = parser.parse_args()
    opt = edict(vars(opt))
    
    # install blender
    print(f'[Rank {opt.rank}/{opt.world_size}] Checking blender...', flush=True)
    _install_blender()

    with open(os.path.join('/'.join(opt.input_dir.split('/')[:-3]), opt.input_dir.split('/')[-2].split('_')[0], 'test_targets_bop19.json'), 'r') as f:
        test_BOP_samples = json.load(f)

    # Create a more efficient data structure: {(scene_id, im_id): set(obj_ids)}
    valid_targets = {}
    for target in test_BOP_samples:
        key = (target['scene_id'], target['im_id'])
        if key not in valid_targets:
            valid_targets[key] = set()
        valid_targets[key].add(target['obj_id'])

    # collect all test cases
    test_cases = sorted(list(set([x.split('.')[0] for x in os.listdir(opt.input_dir)])))
    
    # Distribute test cases across workers
    test_cases = [case for i, case in enumerate(test_cases) if i % opt.world_size == opt.rank]
    print(f'[Rank {opt.rank}/{opt.world_size}] Processing {len(test_cases)} test cases', flush=True)

    # Process test cases in parallel
    records = []
    max_workers = opt.max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor, \
        tqdm(total=len(test_cases), desc=f'[Rank {opt.rank}/{opt.world_size}] Processing cases') as pbar:
        def worker(case):
            result = process_single_case(case, opt, valid_targets)
            pbar.update()
            return result
        
        results = executor.map(worker, test_cases)
        executor.shutdown(wait=True)
        records = [r for r in results if r is not None]
    
    print(f'[Rank {opt.rank}/{opt.world_size}] Completed all {len(test_cases)} test cases', flush=True)
    