"""Idempotently enforce the historical global stride-4 feature protocol."""
from pathlib import Path
import argparse, os
import h5py
import numpy as np

def ensure(path, h5_path, read_only=False):
    path = Path(path)
    with h5py.File(h5_path, 'r') as data:
        full = len(data['pixels'])
    expected = (full + 3) // 4
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    required = ('embeddings', 'actions', 'episode_ids')
    assert all(k in arrays for k in required), 'missing feature arrays'
    rows = len(arrays['embeddings'])
    assert all(len(arrays[k]) == rows for k in required), 'feature row counts disagree'
    assert all(v.ndim == 0 or len(v) == rows for v in arrays.values()), 'non-frame-aligned array'
    if rows == expected:
        return {'rows': rows, 'full_rows': full, 'stride': 4, 'changed': False}
    if rows != full:
        raise ValueError(f'expected full={full} or stride4={expected} rows, got {rows}')
    if read_only:
        raise ValueError('external feature source is full-frame; refusing to change it')
    reduced = {k: (v[::4] if v.ndim else v) for k, v in arrays.items()}
    tmp = path.with_name(path.name + '.stride4.npz')
    np.savez_compressed(tmp, **reduced)
    os.replace(tmp, path)
    return {'rows': expected, 'full_rows': full, 'stride': 4, 'changed': True}

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('features');p.add_argument('h5');p.add_argument('--read-only',action='store_true')
    a=p.parse_args();print('LEGACY_TWOROOM_STRIDE_VERIFIED',ensure(a.features,a.h5,a.read_only),flush=True)
