# This script remeshes a 3D model or a 3D model dataset for subdivision 
# sequence connectivity.
# 
# The MAPS process may fail, due to one of the following reasons:
#   - The base size is too small, and the shape is too complicated. Try to 
#     increase the base size.
#   - There are too much faces in the 3D shape. You may simplify the input
#     mesh before processed.
#   - Nan or other run-time exceptions are encountered, mostly because of 
#     numeric issues. If this happens, you may try to increase the base size, or 
#     enlarge the default value of trial in maps_async to try multiple times.
import os
import trimesh
import numpy as np
import traceback
from maps import MAPS
from multiprocessing import Pool
from multiprocessing.context import TimeoutError as MTE
from pathlib import Path
from tqdm import tqdm


SHREC_CONFIG = {
    'dst_root': './data/SHREC11-MAPS-48-4-split10',
    'src_root': './data/shrec11-split10',
    'n_variation': 10,
    'base_size': 48,
    'depth': 4
}

CUBES_CONFIG = {
    'dst_root': './data/Cubes-MAPS-48-4',
    'src_root': './data/cubes',
    'n_variation': 10,
    'base_size': 48,
    'depth': 4
}

MANIFOLD40_CONFIG = {
    'dst_root': './data/Manifold40-MAPS-96-3',
    'src_root': './data/Manifold40',
    'n_variation': 10,
    'base_size': 96,
    'max_base_size': 192,
    'depth': 3
}


def maps_async(obj_path, out_path, base_size, max_base_size, depth, timeout, 
        trial=1, verbose=False):
    if verbose:
        print('[IN]', out_path)

    for _ in range(trial):
        try:
            mesh = trimesh.load(obj_path, process=False)
            maps = MAPS(mesh.vertices, mesh.faces, base_size, timeout=timeout, 
                verbose=verbose)

            if maps.base_size > max_base_size:
                continue

            sub_mesh = maps.mesh_upsampling(depth=depth)
            sub_mesh.export(out_path)
            break
        except Exception as e:
            if verbose:
                traceback.print_exc()
    else:
        if verbose:
            print('[OUT FAIL]', out_path)
        return False, out_path
    if verbose:
        print('[OUT SUCCESS]', out_path)
    return True, out_path


def make_MAPS_dataset(dst_root, src_root, base_size, depth, n_variation=None, 
        n_worker=1, timeout=None, max_base_size=None, verbose=False):
    '''
    Remeshing a dataset with the MAPS algorithm.

    Parameters
    ----------
    dst_root: str,
        path to a destination directory.
    src_root: str,
        path to the source dataset.
    n_variation:
        number of remeshings for a shape.
    n_workers:
        number of parallel processes.
    timeout:
        if timeout is not None, terminate the MAPS algorithm after timeout seconds.

    References:
        - Lee, Aaron WF, et al. "MAPS: Multiresolution adaptive parameterization of surfaces." 
        Proceedings of the 25th annual conference on Computer graphics and interactive techniques. 1998.
    '''

    if max_base_size is None:
        max_base_size = base_size

    if os.path.exists('maps.log'):
        os.remove('maps.log')

    def callback(pbar, success, path):
        pbar.update()
        if not success:
            with open('maps.log', 'a') as f:
                f.write(str(path) + '\n')    

    for label_dir in sorted(Path(src_root).iterdir(), reverse=True):
        if label_dir.is_dir():
            for mode_dir in sorted(label_dir.iterdir()):
                if mode_dir.is_dir():
                    obj_paths = list(sorted(mode_dir.glob('*.obj')))
                    dst_dir = Path(dst_root) / label_dir.name / mode_dir.name
                    dst_dir.mkdir(parents=True, exist_ok=True)

                    pbar = tqdm(total=len(obj_paths) * n_variation)
                    pbar.set_description(f'{label_dir.name}-{mode_dir.name}')

                    if n_worker > 0:
                        pool = Pool(processes=n_worker)

                    results = []
                    for obj_path in obj_paths:
                        obj_id = str(obj_path.stem)

                        for var in range(n_variation):
                            dst_path = dst_dir / f'{obj_id}-{var}.obj'
                            if dst_path.exists():
                                continue

                            if n_worker > 0:
                                ret = pool.apply_async(
                                    maps_async, 
                                    (str(obj_path), str(dst_path), base_size, max_base_size, depth, timeout), 
                                    callback=lambda x: callback(pbar, x[0], x[1])
                                )
                                results.append(ret)
                            else:
                                maps_async(str(obj_path), str(dst_path), base_size, 
                                    max_base_size, depth, timeout, verbose=verbose)
                                pbar.update()

                    if n_worker > 0:
                        try:
                            [r.get(timeout + 1) for r in results]
                            pool.close()
                        except MTE:
                            pass

                    pbar.close()

def make_MAPS_shape(in_path, out_path, base_size, depth):
    mesh = trimesh.load_mesh(in_path, process=False)
    maps = MAPS(mesh.vertices, mesh.faces, base_size=base_size, verbose=True)
    sub_mesh = maps.mesh_upsampling(depth=depth)
    sub_mesh.export(out_path)


def MAPS_demo1():
    '''Apply MAPS to a single 3D model'''
    make_MAPS_shape('airplane.obj', 'airplane_MAPS.obj', 96, 3)


def MAPS_demo2():
    '''Apply MAPS to shapes from a dataset in parallel'''
    config = MANIFOLD40_CONFIG

    make_MAPS_dataset(
        config['dst_root'], 
        config['src_root'], 
        config['base_size'],
        config['depth'],
        n_variation=config['n_variation'], 
        n_worker=60, 
        timeout=30,
        max_base_size=config.get('max_base_size'),
        verbose=True
    )

if __name__ == "__main__":
    MAPS_demo1()
