import os
import re
import glob
import numpy as np
from typing import List, Optional, Generator, Tuple, Iterable

_CHUNK_NOISY_PATTERN = re.compile(r'kappa_noisy_chunk_(\d+)\.npy')
_CHUNK_LABEL_PATTERN = re.compile(r'label_chunk_(\d+)\.npy')

def get_chunk_indices(chunk_dir: str) -> List[int]:
    """Discover chunk indices that have both noisy and label files."""
    noisy_files = glob.glob(os.path.join(chunk_dir, 'kappa_noisy_chunk_*.npy'))
    label_files = glob.glob(os.path.join(chunk_dir, 'label_chunk_*.npy'))

    noisy_idx = {int(_CHUNK_NOISY_PATTERN.search(os.path.basename(p)).group(1))
                 for p in noisy_files if _CHUNK_NOISY_PATTERN.search(os.path.basename(p))}
    label_idx = {int(_CHUNK_LABEL_PATTERN.search(os.path.basename(p)).group(1))
                 for p in label_files if _CHUNK_LABEL_PATTERN.search(os.path.basename(p))}
    common = sorted(list(noisy_idx & label_idx))
    return common

def _paths_for_index(chunk_dir: str, idx: int) -> Tuple[str, str]:
    noisy_path = os.path.join(chunk_dir, f'kappa_noisy_chunk_{idx}.npy')
    label_path = os.path.join(chunk_dir, f'label_chunk_{idx}.npy')
    return noisy_path, label_path

def load_chunks(chunk_dir: str,
                indices: Optional[Iterable[int]] = None,
                mmap_mode: Optional[str] = None,
                concat_axis: int = 0,
                verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate multiple chunk files into two arrays (noisy, label).
    WARNING: concatenation returns a single array and may use a lot of memory.
    Use iter_chunks or batch_generator for memory-limited workflows.
    """
    if indices is None:
        indices = get_chunk_indices(chunk_dir)
    noisy_list = []
    label_list = []
    for idx in indices:
        noisy_path, label_path = _paths_for_index(chunk_dir, idx)
        if not (os.path.exists(noisy_path) and os.path.exists(label_path)):
            raise FileNotFoundError(f"Missing chunk files for index {idx}: {noisy_path}, {label_path}")
        noisy = np.load(noisy_path, mmap_mode=mmap_mode)
        label = np.load(label_path, mmap_mode=mmap_mode)
        if noisy.shape[0] != label.shape[0]:
            raise ValueError(f"Chunk {idx} sample count mismatch: noisy {noisy.shape[0]} vs label {label.shape[0]}")
        noisy_list.append(np.array(noisy))  # force read for consistent typing if mmap_mode is None
        label_list.append(np.array(label))
        if verbose:
            print(f"Loaded chunk {idx}: noisy {noisy.shape}, label {label.shape}")
    if not noisy_list:
        return np.array([]), np.array([])
    noisy_all = np.concatenate(noisy_list, axis=concat_axis)
    label_all = np.concatenate(label_list, axis=concat_axis)
    return noisy_all, label_all

def iter_chunks(chunk_dir: str,
                indices: Optional[Iterable[int]] = None,
                mmap_mode: Optional[str] = None) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
    """
    Generator that yields (noisy_array, label_array, chunk_idx) for each chunk.
    If mmap_mode is set (e.g. 'r'), the yielded noisy/label may be numpy.memmap objects which support slicing.
    """
    if indices is None:
        indices = get_chunk_indices(chunk_dir)
    for idx in indices:
        noisy_path, label_path = _paths_for_index(chunk_dir, idx)
        if not (os.path.exists(noisy_path) and os.path.exists(label_path)):
            raise FileNotFoundError(f"Missing chunk files for index {idx}: {noisy_path}, {label_path}")
        noisy = np.load(noisy_path, mmap_mode=mmap_mode)
        label = np.load(label_path, mmap_mode=mmap_mode)
        if noisy.shape[0] != label.shape[0]:
            raise ValueError(f"Chunk {idx} sample count mismatch: noisy {noisy.shape[0]} vs label {label.shape[0]}")
        yield noisy, label, idx

def batch_generator(chunk_dir: str,
                    batch_size: int,
                    indices: Optional[Iterable[int]] = None,
                    shuffle: bool = True,
                    shuffle_mode: str = 'per_chunk',  # 'per_chunk' or 'global'
                    mmap_mode: Optional[str] = 'r',
                    seed: Optional[int] = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yield minibatches (noisy_batch, label_batch).
    - per_chunk: iterate chunks (possibly shuffling samples inside each chunk).
    - global: build a global list of (chunk_idx, local_idx) then shuffle and yield batches (requires mmap_mode to avoid heavy memory use).
    mmap_mode is recommended (e.g. 'r') for streaming.
    """
    rng = np.random.default_rng(seed)
    if indices is None:
        indices = get_chunk_indices(chunk_dir)
    indices = list(indices)

    if shuffle_mode not in ('per_chunk', 'global'):
        raise ValueError("shuffle_mode must be 'per_chunk' or 'global'")

    if shuffle_mode == 'per_chunk':
        # Optionally shuffle chunk order
        chunk_order = indices.copy()
        if shuffle:
            rng.shuffle(chunk_order)
        for chunk_idx in chunk_order:
            noisy_path, label_path = _paths_for_index(chunk_dir, chunk_idx)
            noisy = np.load(noisy_path, mmap_mode=mmap_mode)
            label = np.load(label_path, mmap_mode=mmap_mode)
            n = noisy.shape[0]
            order = np.arange(n)
            if shuffle:
                rng.shuffle(order)
            # yield batches within this chunk
            for start in range(0, n, batch_size):
                batch_idx = order[start:start + batch_size]
                yield noisy[batch_idx], label[batch_idx]

    else:  # global shuffle across all chunks
        # Build mapping (chunk_idx, local_idx). This can be large but stores only pairs of ints.
        index_map = []
        lengths = {}
        for chunk_idx in indices:
            noisy_path, label_path = _paths_for_index(chunk_dir, chunk_idx)
            noisy = np.load(noisy_path, mmap_mode=mmap_mode)
            label = np.load(label_path, mmap_mode=mmap_mode)
            if noisy.shape[0] != label.shape[0]:
                raise ValueError(f"Chunk {chunk_idx} sample count mismatch: noisy {noisy.shape[0]} vs label {label.shape[0]}")
            n = noisy.shape[0]
            lengths[chunk_idx] = n
            index_map.extend((chunk_idx, i) for i in range(n))
        if shuffle:
            rng.shuffle(index_map)
        # open memmaps and keep in cache to avoid repeated disk opens
        memmaps = {}
        for start in range(0, len(index_map), batch_size):
            batch_pairs = index_map[start:start + batch_size]
            # group by chunk for efficient slicing
            groups = {}
            for cidx, lidx in batch_pairs:
                groups.setdefault(cidx, []).append(lidx)
            batch_noisy_parts = []
            batch_label_parts = []
            for cidx, local_idxs in groups.items():
                if cidx not in memmaps:
                    noisy_path, label_path = _paths_for_index(chunk_dir, cidx)
                    memmaps[cidx] = (np.load(noisy_path, mmap_mode=mmap_mode),
                                     np.load(label_path, mmap_mode=mmap_mode))
                noisy_mem, label_mem = memmaps[cidx]
                idx_arr = np.array(local_idxs, dtype=np.intp)
                batch_noisy_parts.append(noisy_mem[idx_arr])
                batch_label_parts.append(label_mem[idx_arr])
            # concatenate order must match batch_pairs order
            # Reconstruct batch in correct order:
            batch_noisy = []
            batch_label = []
            for cidx, lidx in batch_pairs:
                # find the position inside groups[cidx] to extract the corresponding row
                # faster approach: use dict[(cidx, lidx)] -> position, but for small batch sizes this is fine:
                # find index of lidx in groups[cidx] (but groups[cidx] may have duplicates; use pop approach)
                pass
            # Simpler: gather by iterating batch_pairs and indexing memmaps
            batch_noisy = np.stack([memmaps[cidx][0][lidx] for cidx, lidx in batch_pairs], axis=0)
            batch_label = np.stack([memmaps[cidx][1][lidx] for cidx, lidx in batch_pairs], axis=0)
            yield batch_noisy, batch_label