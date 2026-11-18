from pathlib import Path

import numpy as np


def side_path(cache_dir, pid, name):
    return Path(cache_dir) / "_side" / f"{pid}__{name}.npz"


def has_side(cache_dir, pid, name):
    return side_path(cache_dir, pid, name).exists()


def save_side(cache_dir, pid, name, arr):
    p = side_path(cache_dir, pid, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(p), a=np.asarray(arr))


def load_side(cache_dir, pid, name):
    p = side_path(cache_dir, pid, name)
    return np.load(str(p))["a"] if p.exists() else None
