"""Raw array-order cross-sections with independent, cancellable file readers."""
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np

from .data import read_image, validate_frame


@dataclass(frozen=True)
class SequenceSpec:
    paths: tuple
    indices: tuple
    shape: tuple
    kind: str
    dtype: str
    array_shape: tuple | None
    pages: bool
    storage: object = field(default=None, compare=False, repr=False)
    volume_cache: object = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dataset(cls, dataset, *, cache_sections=False):
        if cache_sections and dataset._section_cache is None:
            dataset._section_cache = SectionVolumeCache(dataset.dtype, dataset._temporary_storage)
        return cls(dataset.paths, dataset.indices, dataset.shape, dataset.kind,
                   dataset._raw_dtype.str, tuple(dataset._array.shape) if dataset._array is not None else None,
                   dataset._pages, getattr(dataset, '_temporary_storage', None),
                   dataset._section_cache if cache_sections else None)


class SectionVolumeCache:
    """Native-dtype temporary volume, shared by panes and resumable on cancel."""
    def __init__(self, dtype, storage=None):
        self.dtype = np.dtype(dtype)
        self.directory = None
        self.path = None
        self.array = None
        self.storage = storage
        self.completed = 0
        self.errors = {}
        self.disabled = False
        self.lock = Lock()

    def prepare(self, spec, cancelled, progress):
        while not self.lock.acquire(timeout=.05):
            if cancelled.is_set():
                return None
        try:
            if cancelled.is_set() or self.disabled:
                return None
            if self.completed == len(spec.indices):
                return self.array, dict(self.errors)
            if spec.array_shape is not None:
                self.array = (np.load(spec.paths[0], mmap_mode='r', allow_pickle=False)
                              if spec.kind == 'array' else
                              np.memmap(spec.paths[0], mode='r', dtype=spec.dtype, shape=spec.array_shape))
                self.completed = len(spec.indices)
                return self.array, {}
            if self.directory is None:
                self.directory = TemporaryDirectory(prefix='metal-review-sections-')
                self.path = Path(self.directory.name) / 'volume.npy'
                self.array = np.lib.format.open_memmap(self.path, mode='w+', dtype=self.dtype,
                                                       shape=(len(spec.indices), *spec.shape))
            mapped = self.array
            for position in range(self.completed, len(spec.indices)):
                if cancelled.is_set():
                    return None
                number = spec.indices[position]
                try:
                    frame = (np.fromfile(spec.paths[position], dtype=spec.dtype).reshape(spec.shape)
                             if spec.kind == 'raw' else
                             read_image(spec.paths[0] if spec.pages else spec.paths[position],
                                        position if spec.pages else 0))
                    validate_frame(frame)
                    if tuple(frame.shape) != tuple(spec.shape):
                        raise ValueError(f'尺寸 {frame.shape} 与首层 {spec.shape} 不一致')
                except Exception as error:
                    self.errors[number] = f'层 {number}：{error}'
                    mapped[position] = 0
                else:
                    if not np.can_cast(frame.dtype, self.dtype, casting='safe'):
                        # Mixed-dtype image sequences retain the original reader,
                        # rather than silently clipping values to the first dtype.
                        self.disabled = True
                        return None
                    mapped[position] = frame
                self.completed = position + 1
                if progress and (position % 20 == 0 or self.completed == len(spec.indices)):
                    progress(self.completed, len(spec.indices))
            return self.array, dict(self.errors)
        except (OSError, MemoryError):
            # Lack of temporary disk space must not prevent ordinary inspection.
            self.disabled = True
            return None
        finally:
            try:
                if self.disabled:
                    if self.array is not None:
                        self.array._mmap.close()
                        self.array = None
                    if self.directory is not None:
                        self.directory.cleanup()
                        self.directory = None
                        self.path = None
            finally:
                self.lock.release()

    def __del__(self):
        if self.array is not None:
            self.array._mmap.close()
        if self.directory is not None:
            self.directory.cleanup()
        self.storage = None


def section_indices(indices):
    """Leave ordinary missing layers blank; avoid huge allocations for sparse IDs."""
    low, high = min(indices), max(indices)
    return tuple(range(low, high + 1)) if high - low + 1 <= max(4096, 4 * len(indices)) else tuple(sorted(indices))


def sections_from_array(array, spec, row, column, cancelled, progress, errors=None):
    """Extract both planes directly; missing/bad frame IDs remain blank."""
    indices = section_indices(spec.indices)
    positions = {number: i for i, number in enumerate(indices)}
    channels = spec.shape[2:]
    dtype, fill = (np.uint8, 0) if channels else (np.float64, np.nan)
    rows = np.full((len(indices), spec.shape[1], *channels), fill, dtype=dtype)
    columns = np.full((len(indices), spec.shape[0], *channels), fill, dtype=dtype)
    errors = errors or {}
    valid = [i for i, number in enumerate(spec.indices) if number not in errors]
    destinations = [positions[spec.indices[i]] for i in valid]
    volume = array[None] if array.ndim == len(spec.shape) else array
    rows[destinations] = volume[valid, row, ...]
    columns[destinations] = volume[valid, :, column, ...]
    if cancelled.is_set():
        return None
    if progress:
        progress(len(spec.indices), len(spec.indices))
    return rows, columns, indices, list(errors.values())


def read_sections(spec, row, column, cancelled, progress=None):
    """Return D[:,row,:], D[:,:,column] without loading an entire volume.

    Workers retain cache mappings or own their standalone source mmap, so
    clearing GUI datasets cannot close a mapping still in use. No flips or
    transposes are applied.
    """
    if cancelled.is_set():
        return None
    height, width = spec.shape[:2]
    if not 0 <= row < height or not 0 <= column < width:
        raise ValueError('三视图的行、列索引越界')
    if spec.volume_cache is not None:
        prepared = spec.volume_cache.prepare(spec, cancelled, progress)
        if cancelled.is_set():
            return None
        if prepared is not None:
            cached, errors = prepared
            return sections_from_array(cached, spec, row, column, cancelled, progress, errors)
    indices = section_indices(spec.indices)
    positions = {number: position for position, number in enumerate(spec.indices)}
    channels = spec.shape[2:]
    dtype = np.uint8 if channels else np.float64
    fill = 0 if channels else np.nan
    rows = np.full((len(indices), width, *channels), fill, dtype=dtype)
    columns = np.full((len(indices), height, *channels), fill, dtype=dtype)
    errors = []
    array = None
    try:
        if spec.array_shape is not None:
            if spec.kind == 'array':
                array = np.load(spec.paths[0], mmap_mode='r', allow_pickle=False)
            else:
                array = np.memmap(spec.paths[0], mode='r', dtype=spec.dtype, shape=spec.array_shape)
            return sections_from_array(array, spec, row, column, cancelled, progress)
        for output, number in enumerate(indices):
            if cancelled.is_set():
                return None
            position = positions.get(number)
            if position is None:
                continue
            frame = None
            mapped = None
            try:
                if array is not None:
                    frame = array if array.ndim == 2 else array[position]
                elif spec.kind == 'raw':
                    mapped = np.memmap(spec.paths[position], mode='r', dtype=spec.dtype, shape=spec.shape)
                    frame = mapped
                else:
                    frame = read_image(spec.paths[0] if spec.pages else spec.paths[position],
                                       position if spec.pages else 0)
                if tuple(frame.shape) != tuple(spec.shape):
                    raise ValueError(f'尺寸 {frame.shape} 与首层 {spec.shape} 不一致')
                rows[output] = frame[row, ...]
                columns[output] = frame[:, column, ...]
            except Exception as error:
                errors.append(f'层 {number}：{error}')
            finally:
                frame = None
                if mapped is not None:
                    mapped._mmap.close()
            if progress is not None and (output % 20 == 0 or output + 1 == len(indices)):
                progress(output + 1, len(indices))
        return rows, columns, indices, errors
    finally:
        if array is not None:
            array._mmap.close()
