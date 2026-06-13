import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict
from functools import partial
from subprocess import DEVNULL, call


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


def _convert(file_path, sha256, output_dir):
    output_folder = os.path.join(output_dir, 'converted_meshes', sha256)

    args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'blender_script', 'convert.py'),
        '--',
        '--object', os.path.expanduser(file_path),
        '--output_folder', output_folder,
        '--save_mesh',
    ]
    if file_path.endswith('.blend'):
        args.insert(1, file_path)

    os.makedirs(output_folder, exist_ok=True)
    log_file = os.path.join(output_folder, 'convert.log')
    with open(log_file, 'w') as log:
        call(args, stdout=log, stderr=log)


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=5.5,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=8)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))

    os.makedirs(os.path.join(opt.output_dir, 'converted_meshes'), exist_ok=True)

    # install blender
    print('Checking blender...', flush=True)
    _install_blender()

    # get file list
    if not os.path.exists(os.path.join(opt.output_dir, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.output_dir, 'metadata.csv'))
    if opt.instances is None:
        metadata = metadata[metadata['local_path'].notna()]
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        # if 'rendered' in metadata.columns:
        #     metadata = metadata[metadata['rendered'] == False]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    # filter out objects that are already processed
    for sha256 in copy.copy(metadata['sha256'].values):
        if os.path.exists(os.path.join(opt.output_dir, 'converted_meshes', sha256, 'mesh.glb')):
            # records.append({'sha256': sha256, 'rendered': True})
            metadata = metadata[metadata['sha256'] != sha256]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]

    print(f'Processing {len(metadata)} objects...')

    # process objects
    func = partial(_convert, output_dir=opt.output_dir)
    dataset_utils.foreach_instance(metadata, opt.output_dir, func, max_workers=opt.max_workers, desc='Converting objects')
