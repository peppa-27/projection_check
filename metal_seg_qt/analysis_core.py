"""Exact mask support counts and calibrated reprojection diagnostics."""
from dataclasses import replace
from pathlib import Path

import numpy as np
from numba import njit, prange

from .data import Dataset, ImportOptions
from .processing import TemporaryMaskStorage, check_cancelled, sequence_frames


@njit(cache=True, parallel=True)
def count_chunk(masks, matrices, start, depth, height, width):
    hits = np.zeros((depth, height, width), np.uint32)
    valid = np.zeros((depth, height, width), np.uint32)
    for task in prange(depth * height):
        z, y = start + task // height, task % height
        for x in range(width):
            h, n = 0, 0
            for i in range(len(matrices)):
                p = matrices[i]
                w = p[2, 0]*x + p[2, 1]*y + p[2, 2]*z + p[2, 3]
                if w <= 1e-12:
                    continue
                u = (p[0, 0]*x + p[0, 1]*y + p[0, 2]*z + p[0, 3]) / w
                v = (p[1, 0]*x + p[1, 1]*y + p[1, 2]*z + p[1, 3]) / w
                if -.5 <= u < masks.shape[2] - .5 and -.5 <= v < masks.shape[1] - .5:
                    n += 1
                    h += masks[i, int(np.floor(v+.5)), int(np.floor(u+.5))] != 0
            hits[task // height, y, x] = h
            valid[task // height, y, x] = n
    return hits, valid


def sampled_geometry(geometry, factor):
    if factor not in (1, 2, 4, 8):
        raise ValueError('预览采样倍数无效')
    native = np.array([geometry.slice_shape[1], geometry.slice_shape[0],
                       geometry.slice_end - geometry.slice_begin + 1])
    counts = (native + factor - 1) // factor
    scale = native / counts
    basis = np.column_stack((geometry.column_step, geometry.row_step, geometry.slice_step))
    origin = geometry.origin + basis @ ((scale - 1) / 2)
    begin = geometry.slice_begin if factor == 1 else 0
    return replace(geometry, slice_shape=(int(counts[1]), int(counts[0])),
        slice_begin=begin, slice_end=begin + int(counts[2]) - 1, origin=origin,
        column_step=geometry.column_step * scale[0], row_step=geometry.row_step * scale[1],
        slice_step=geometry.slice_step * scale[2],
        coordinate_convention=geometry.coordinate_convention + (f' · 预览采样 ×{factor}' if factor > 1 else ''))


def local_matrices(geometry, indices, image_shape):
    h, w = image_shape
    dh, dw = geometry.detector_shape
    if not np.isclose(w / h, dw / dh, rtol=1e-3):
        raise ValueError('投影掩膜比例与探测器不符，无法推测裁剪或拼接关系')
    positions = np.array(indices) - geometry.projection_start
    if (positions < 0).any() or (positions >= len(geometry.matrices)).any():
        raise ValueError('输入掩膜的真实帧号超出 PTM 范围')
    affine = np.eye(4)
    affine[:3, :3] = np.column_stack((geometry.column_step, geometry.row_step, geometry.slice_step))
    affine[:3, 3] = geometry.origin
    sx, sy = w / dw, h / dh
    detector = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
    return np.ascontiguousarray(detector @ geometry.matrices[positions] @ affine)


class SupportResult:
    def __init__(self, source, geometry, factor):
        self.source = source
        self.native_geometry = geometry
        self.geometry = sampled_geometry(geometry, factor)
        self.factor = factor
        self.indices = tuple(sorted(source.sequence.indices))
        self.positions = {number: i for i, number in enumerate(self.indices)}
        self.shape = (self.geometry.slice_end-self.geometry.slice_begin+1, *self.geometry.slice_shape)
        self.matrices = local_matrices(self.geometry, self.indices, source.sequence.shape)
        self.storage = TemporaryMaskStorage('metal-analysis-support-')
        self.arrays = []
        self.paths = {}

    def allocate(self, name, dtype, shape):
        path = Path(self.storage.name) / f'{name}.npy'
        array = np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)
        self.arrays.append(array)
        self.paths[name] = path
        return array

    def close(self):
        for a in self.arrays:
            a._mmap.close()
        self.arrays.clear()
        self.storage = None

    def __del__(self):
        if hasattr(self, 'arrays'):
            self.close()

    @property
    def voxel_mm3(self):
        g = self.geometry
        return abs(float(np.linalg.det(np.column_stack((g.column_step, g.row_step, g.slice_step)))))

    def votes(self, xyz):
        x, y, z = xyz
        p = self.matrices
        points = p[:, :, 0]*x + p[:, :, 1]*y + p[:, :, 2]*z + p[:, :, 3]
        good = points[:, 2] > 1e-12
        uv = np.zeros((len(p), 2))
        np.divide(points[:, :2], points[:, 2:3], out=uv, where=good[:, None])
        h, w = self.masks.shape[1:]
        good &= ((uv[:, 0] >= -.5) & (uv[:, 0] < w-.5) &
                 (uv[:, 1] >= -.5) & (uv[:, 1] < h-.5))
        states = np.full(len(p), -1, np.int8)
        ids = np.flatnonzero(good)
        pixels = np.floor(uv[ids]+.5).astype(int)
        states[ids] = self.masks[ids, pixels[:, 1], pixels[:, 0]]
        return states, uv, good


def build_support(source, geometry, factor, cancelled, progress=None):
    if source.domain != 'projection' or not source.mask or len(source.sequence.shape) != 2:
        raise ValueError('请选择投影域二值掩膜，普通投影图像不能直接用于一致性分析')
    result = SupportResult(source, geometry, factor)
    try:
        check_cancelled(cancelled)
        result.masks = result.allocate('input', np.uint8, (len(result.indices), *source.sequence.shape))
        with sequence_frames(source.sequence) as frames:
            for i, (number, frame) in enumerate(frames):
                check_cancelled(cancelled)
                if not np.isin(frame, [0, 1, 255, 65535]).all():
                    raise ValueError(f'投影 {number} 含非二值像素；请检查掩膜类型和 Raw dtype')
                result.masks[result.positions[number]] = frame != 0
                if progress and (i % 20 == 0 or i+1 == len(result.indices)):
                    progress(f'读取输入掩膜 {i+1}/{len(result.indices)}')
        dtype = np.uint16 if len(result.indices) <= 65535 else np.uint32
        result.hits = result.allocate('hits', dtype, result.shape)
        result.valid = result.allocate('valid', dtype, result.shape)
        result.score = result.allocate('support', np.float32, result.shape)
        if progress:
            progress('统计三维支持率… 首次使用需要编译计算内核')
        for start in range(0, result.shape[0], 2):
            check_cancelled(cancelled)
            depth = min(2, result.shape[0]-start)
            h, v = count_chunk(result.masks, result.matrices, start, depth, *result.shape[1:])
            result.hits[start:start+depth] = h
            result.valid[start:start+depth] = v
            score = np.full(h.shape, np.nan, np.float32)
            np.divide(h, v, out=score, where=v > 0)
            result.score[start:start+depth] = score
            if progress:
                progress(f'统计三维支持率 {start+depth}/{result.shape[0]} 层 · 全部 {len(result.indices)} 个输入视角')
        check_cancelled(cancelled)
        return result
    except BaseException:
        result.close()
        raise


def envelope(result, threshold, min_views, cancelled=None):
    # Compare counts in float64; the persisted display score is float32.
    # Slabs avoid two volume-sized float64 temporaries on native grids.
    volume = np.empty(result.shape, bool)
    for start in range(0, result.shape[0], 8):
        if cancelled is not None:
            check_cancelled(cancelled)
        valid = result.valid[start:start+8]
        volume[start:start+8] = ((valid >= min_views) & (valid > 0) &
            (result.hits[start:start+8].astype(np.float64)+1e-9 >= threshold*valid))
    return volume


def difference_counts(a, b, roi=None):
    if roi is not None:
        h, w = a.shape
        x0, y0, x1, y1 = np.clip(roi, 0, 1)
        left = min(w-1, max(0, int(min(x0, x1)*w)))
        top = min(h-1, max(0, int(min(y0, y1)*h)))
        right = min(w, max(left+1, int(np.ceil(max(x0, x1)*w))))
        bottom = min(h, max(top+1, int(np.ceil(max(y0, y1)*h))))
        a, b = a[top:bottom, left:right], b[top:bottom, left:right]
    a, b = a != 0, b != 0
    return np.array([np.count_nonzero(a & b), np.count_nonzero(a & ~b),
                     np.count_nonzero(~a & b)], np.int64)


def difference_pixels(a, b):
    a, b = a != 0, b != 0
    pixels = np.zeros((*a.shape, 3), np.uint8)
    pixels[a & b] = [192, 203, 213]
    pixels[a & ~b] = [255, 166, 69]
    pixels[~a & b] = [193, 117, 255]
    return pixels


class ProjectionSeries:
    def __init__(self, source, threshold, min_views):
        self.source, self.threshold, self.min_views = source, threshold, min_views
        self.storage = TemporaryMaskStorage('metal-analysis-reprojection-')
        self.path = Path(self.storage.name) / 'reprojection.npy'
        self.array = np.lib.format.open_memmap(self.path, mode='w+', dtype=np.uint8, shape=source.masks.shape)
        self.ready = np.zeros(len(source.indices), bool)
        self.counts = np.zeros((len(source.indices), 3), np.int64)
        self.volume_count = 0

    def __del__(self):
        if hasattr(self, 'array'):
            self.array._mmap.close()
        self.storage = None


def projection_setup(volume, matrices):
    active = [np.flatnonzero(volume.any(axis=axes)) for axes in ((0, 1), (0, 2), (1, 2))]
    if any(not len(a) for a in active):
        return None
    lo = np.array([a[0] for a in active])
    hi = np.array([a[-1]+1 for a in active])
    lower, upper = lo-.5, hi-.5
    cropped = np.ascontiguousarray(volume[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]])
    inverse = np.linalg.inv(matrices[:, :, :3])
    sources = -np.einsum('nij,nj->ni', inverse, matrices[:, :, 3])
    corners = np.array([[x, y, z, 1.] for x in (lower[0], upper[0])
                       for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    projected = np.einsum('nij,kj->nki', matrices, corners)
    return cropped, inverse, sources, lower, projected


def reproject(result, threshold, min_views, cancelled, progress=None, partial=None, priority=None):
    from consistency.consistency_filter import _forward
    check_cancelled(cancelled)
    if not 0 <= threshold <= 1 or not 1 <= min_views <= len(result.indices):
        raise ValueError('一致性阈值或最少有效视角数无效')
    series = ProjectionSeries(result, threshold, min_views)
    volume = envelope(result, threshold, min_views, cancelled)
    series.volume_count = int(np.count_nonzero(volume))
    prepared = projection_setup(volume, result.matrices)
    if partial:
        partial(('series', series))
    pending = list(range(len(result.indices)))
    height, width = result.masks.shape[1:]
    while pending:
        check_cancelled(cancelled)
        requested = priority() if priority else None
        index = requested if requested in pending else pending[0]
        pending.remove(index)
        if prepared is None:
            series.array[index] = 0
        else:
            cropped, inverse, sources, lower, projected = prepared
            bounds = np.array([[0, width, 0, height]], np.int64)
            points = projected[index]
            if (points[:, 2] > 1e-12).all():
                uv = points[:, :2] / points[:, 2:3]
                bounds[0] = [np.clip(np.floor(uv[:, 0].min())-1, 0, width),
                             np.clip(np.ceil(uv[:, 0].max())+2, 0, width),
                             np.clip(np.floor(uv[:, 1].min())-1, 0, height),
                             np.clip(np.ceil(uv[:, 1].max())+2, 0, height)]
            series.array[index] = _forward(cropped, inverse[index:index+1], sources[index:index+1],
                                            bounds, lower, np.ones(3), height, width)[0]
        series.counts[index] = difference_counts(result.masks[index], series.array[index])
        series.ready[index] = True
        if partial:
            partial(('view', index))
        if progress:
            progress(f'重投影 {len(result.indices)-len(pending)}/{len(result.indices)} · τ={threshold:.3f}')
    check_cancelled(cancelled)
    return series


def roi_metrics(series, roi, cancelled):
    counts = np.full((len(series.ready), 3), -1, np.int64)
    for index in np.flatnonzero(series.ready.copy()):
        check_cancelled(cancelled)
        counts[index] = difference_counts(series.source.masks[index], series.array[index], roi)
    return counts


def ray_profile(result, ray):
    """DDA cell intervals along the positive, already clipped calibrated ray."""
    a, b = ray.local_endpoints
    delta = b-a
    counts = np.array(result.shape[::-1])
    index = np.floor(a+.5 + np.sign(delta)*1e-9).astype(int)
    index = np.clip(index, 0, counts-1)
    step = np.sign(delta).astype(int)
    next_t, dt = np.full(3, np.inf), np.full(3, np.inf)
    for axis in range(3):
        if step[axis]:
            boundary = index[axis] + (.5 if step[axis] > 0 else -.5)
            next_t[axis] = (boundary-a[axis]) / delta[axis]
            dt[axis] = 1/abs(delta[axis])
    cells, distance, score = [], [], []
    length = float(np.linalg.norm(ray.world_endpoints[1]-ray.world_endpoints[0]))
    if length <= 1e-12:
        return np.array([]), np.array([]), np.empty((0, 3), dtype=int)
    t = 0.
    for _ in range(int(counts.sum())+3):
        end = min(1., next_t.min())
        if end > t+1e-12:
            x, y, z = index
            cells.append(index.copy())
            distance.append((t+end)/2*length)
            score.append(float(result.score[z, y, x]))
        if end >= 1:
            break
        crossed = next_t <= end+1e-12
        index[crossed] += step[crossed]
        next_t[crossed] += dt[crossed]
        if (index < 0).any() or (index >= counts).any():
            break
        t = end
    return np.array(distance), np.array(score), np.array(cells, dtype=int).reshape(-1, 3)


def result_dataset(source, path, storage, name, domain, mask, description, geometry=None, indices=None):
    dataset = Dataset([path], ImportOptions(domain=domain, mask=mask, use_xml=False))
    dataset._temporary_storage = storage
    dataset.generated = True
    dataset.geometry_paths = source.source.geometry_paths
    dataset.name, dataset.description = name, description
    if geometry is not None:
        dataset.geometry_override = geometry
    dataset.indices = tuple(indices)
    dataset._positions = {n: i for i, n in enumerate(dataset.indices)}
    dataset._cache.clear()
    dataset.numbered = True
    dataset.shape_source = description
    return dataset


def export_support(result):
    g = result.geometry
    return result_dataset(result, result.paths['support'], result.storage,
        result.source.name+' · 支持率体', 'reconstruction', False,
        f'支持率 0–1；未知为 NaN；采样 ×{result.factor}，保留实际物理网格', g,
        range(g.slice_begin, g.slice_end+1))


def export_envelope(result, threshold, min_views, cancelled):
    storage = TemporaryMaskStorage('metal-analysis-envelope-')
    path = Path(storage.name) / 'envelope.npy'
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=result.shape)
    try:
        for z in range(result.shape[0]):
            check_cancelled(cancelled)
            array[z] = ((result.valid[z] >= min_views) & (result.valid[z] > 0) &
                        (result.hits[z].astype(np.float64)+1e-9 >= threshold*result.valid[z]))
    finally:
        array._mmap.close()
    check_cancelled(cancelled)
    g = result.geometry
    return result_dataset(result, path, storage, f'{result.source.name} · 包络 τ={threshold:.3f}',
        'reconstruction', True, f'二值支持包络；τ={threshold:g}；有效视角≥{min_views}；采样 ×{result.factor}',
        g, range(g.slice_begin, g.slice_end+1))


def export_reprojection(series):
    if not series.ready.all():
        raise ValueError('重投影尚未完成，请等待全部视角计算结束')
    return result_dataset(series.source, series.path, series.storage,
        f'{series.source.source.name} · 一致性重投影 τ={series.threshold:.3f}',
        'projection', True, f'包络重投影；τ={series.threshold:g}；有效视角≥{series.min_views}；采样 ×{series.source.factor}',
        indices=series.source.indices)
