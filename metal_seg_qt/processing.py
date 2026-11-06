"""Cancellable, whole-sequence mask operations; source images stay read-only."""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .data import Dataset, ImportOptions, read_image, validate_frame
from .orthogonal import SequenceSpec


class ProcessingCancelled(Exception):
    pass


class TemporaryMaskStorage:
    """Shared by a dataset and reader snapshots; cleanup follows the last owner."""
    def __init__(self, prefix):
        self.directory = None
        self.directory = TemporaryDirectory(prefix=prefix)

    @property
    def name(self):
        return self.directory.name

    def cleanup(self):
        if self.directory is not None:
            self.directory.cleanup()

    def __del__(self):
        self.cleanup()


@dataclass(frozen=True)
class ProcessingSource:
    sequence: SequenceSpec
    name: str
    domain: str
    mask: bool
    geometry_paths: tuple
    geometry_override: object = None

    @classmethod
    def from_dataset(cls, dataset):
        return cls(SequenceSpec.from_dataset(dataset), dataset.name, dataset.domain,
                   dataset.mask, dataset.geometry_paths, getattr(dataset, 'geometry_override', None))


def check_cancelled(cancelled):
    if cancelled.is_set():
        raise ProcessingCancelled()


@contextmanager
def sequence_frames(spec):
    """Own the source mapping independently of the visible/removed dataset."""
    array = None
    try:
        if spec.array_shape is not None:
            array = (np.load(spec.paths[0], mmap_mode='r', allow_pickle=False)
                     if spec.kind == 'array' else
                     np.memmap(spec.paths[0], mode='r', dtype=spec.dtype, shape=spec.array_shape))

        def frames():
            for position, number in enumerate(spec.indices):
                if array is not None:
                    frame = array if array.ndim == 2 else array[position]
                elif spec.kind == 'raw':
                    frame = np.fromfile(spec.paths[position], dtype=spec.dtype).reshape(spec.shape)
                else:
                    frame = read_image(spec.paths[0] if spec.pages else spec.paths[position],
                                       position if spec.pages else 0)
                validate_frame(frame)
                if tuple(frame.shape) != tuple(spec.shape):
                    raise ValueError(f'层 {number} 尺寸不一致，已停止处理')
                yield number, frame

        yield frames()
    finally:
        if array is not None:
            array._mmap.close()


def finish_mask(storage, path, source, indices, domain, name, description):
    result = Dataset([path], ImportOptions(domain=domain, mask=True, use_xml=False))
    result._temporary_storage = storage
    result.generated = True
    result.geometry_paths = source.geometry_paths
    if domain == 'reconstruction' and source.geometry_override is not None:
        result.geometry_override = source.geometry_override
    result.name, result.description = name, description
    result.indices = tuple(indices)
    result._positions = {number: i for i, number in enumerate(result.indices)}
    result._cache.clear()
    result.numbered = True
    result.shape_source = '处理结果自带尺寸；保留来源几何与真实层号'
    return result


def binary_threshold(frame, threshold, comparison):
    """Shared rule for live previews and full-sequence mask generation."""
    if not np.isfinite(threshold) or comparison not in ('ge', 'le'):
        raise ValueError('阈值必须是有限数值，比较方式须为 ≥ 或 ≤')
    passed = frame >= threshold if comparison == 'ge' else frame <= threshold
    return np.isfinite(frame) & passed


def threshold_mask(source, threshold, comparison, cancelled, progress=None):
    if source.domain != 'reconstruction' or source.mask or len(source.sequence.shape) != 2:
        raise ValueError('阈值分割需要重建域的灰度图像，请选择重建原图')
    if not np.isfinite(threshold) or comparison not in ('ge', 'le'):
        raise ValueError('阈值必须是有限数值')
    check_cancelled(cancelled)
    storage = TemporaryMaskStorage(prefix='metal-review-threshold-')
    path = Path(storage.name) / 'mask.npy'
    output = None
    try:
        spec = source.sequence
        output = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                          shape=(len(spec.indices), *spec.shape))
        with sequence_frames(spec) as frames:
            for position, (number, frame) in enumerate(frames):
                check_cancelled(cancelled)
                output[position] = binary_threshold(frame, threshold, comparison)
                if progress:
                    progress(f'阈值分割：{position + 1}/{len(spec.indices)} 层')
        output.flush()
        output._mmap.close()
        output = None
        check_cancelled(cancelled)
        operator = '≥' if comparison == 'ge' else '≤'
        rule = f'{operator}{threshold:g}'
        result = finish_mask(storage, path, source, spec.indices, 'reconstruction',
                             f'{source.name} · 阈值{rule}',
                             f'重建阈值掩膜：原始像素值 {rule}，有限值参与；整组切片，前景 1 / 背景 0。')
        storage = None
        return result
    finally:
        if output is not None:
            output._mmap.close()
        if storage is not None:
            storage.cleanup()


def project_mask(source, geometry, cancelled, progress=None):
    if source.domain != 'reconstruction' or not source.mask or len(source.sequence.shape) != 2:
        raise ValueError('只能将重建域二值掩膜生成投影掩膜，不能使用普通重建图像')
    spec = source.sequence
    expected = tuple(range(geometry.slice_begin, geometry.slice_end + 1))
    if tuple(sorted(spec.indices)) != expected:
        raise ValueError('重建掩膜必须覆盖几何文件中的全部切片；当前有缺层，无法完整生成投影')
    check_cancelled(cancelled)
    try:
        from consistency.consistency_filter import VolumeGrid, forward_project_envelope
    except ImportError as error:
        raise ValueError('掩膜投影需要 numba，请在当前环境安装 metal_seg_qt/requirements.txt') from error
    storage = TemporaryMaskStorage(prefix='metal-review-projection-')
    path = Path(storage.name) / 'mask.npy'
    volume_path = Path(storage.name) / 'volume.npy'
    volume = output = None
    try:
        volume = np.lib.format.open_memmap(volume_path, mode='w+', dtype=np.bool_,
                                          shape=(len(expected), *spec.shape))
        with sequence_frames(spec) as frames:
            for position, (number, frame) in enumerate(frames):
                check_cancelled(cancelled)
                # The sidebar mask checkbox is a display hint, not permission
                # to binarize arbitrary CT intensities implicitly.
                if not np.isin(frame, [0, 1, 255, 65535]).all():
                    raise ValueError(f'层 {number} 含非二值掩膜像素；请先对重建原图进行阈值分割')
                volume[number - geometry.slice_begin] = frame != 0
                if progress:
                    progress(f'读取重建掩膜：{position + 1}/{len(expected)} 层')
        volume.flush()
        # Transform the calibrated PTM into native voxel coordinates, so DDA
        # also handles signed axes and oblique INI bases without resampling.
        voxel_to_world = np.eye(4)
        voxel_to_world[:3, :3] = np.column_stack((geometry.column_step, geometry.row_step,
                                                 geometry.slice_step))
        voxel_to_world[:3, 3] = geometry.origin
        matrices = geometry.matrices @ voxel_to_world
        grid = VolumeGrid(volume.shape, (1., 1., 1.), (-.5, -.5, -.5))
        output = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                          shape=(len(matrices), *geometry.detector_shape))

        def report(message):
            check_cancelled(cancelled)
            if progress:
                if message.startswith('Forward projection:') and 'views' in message:
                    count = message.split(':', 1)[1].strip().split()[0]
                    progress(f'生成投影掩膜：{count} 个角度')
                else:
                    progress('正在计算掩膜投影（首次运行需要编译）…')

        report('start')
        forward_project_envelope(volume, matrices, grid, geometry.detector_shape,
                                 progress=report, out=output, batch_size=4)
        check_cancelled(cancelled)
        output.flush()
        output._mmap.close()
        output = None
        volume._mmap.close()
        volume = None
        volume_path.unlink()
        indices = range(geometry.projection_start, geometry.projection_start + len(matrices))
        result = finish_mask(storage, path, source, indices, 'projection',
                             f'{source.name} · 投影掩膜',
                             '重建掩膜的二值射线投影：射线穿过前景体素则为 1，否则为 0；'
                             '使用同样本 PTM 的全部角度与原始探测器尺寸。')
        storage = None
        return result
    finally:
        for mapped in (volume, output):
            if mapped is not None:
                mapped._mmap.close()
        if storage is not None:
            storage.cleanup()
