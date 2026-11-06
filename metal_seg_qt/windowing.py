"""Reusable raw-intensity percentiles for experimental display windowing."""
from dataclasses import dataclass

import numpy as np

from .processing import check_cancelled, sequence_frames


@dataclass
class IntensityStatistics:
    values: np.ndarray
    counts: np.ndarray | None = None
    frames: int = 1
    finite_count: int = 0
    approximate: bool = False

    @classmethod
    def from_frame(cls, frame):
        if frame.ndim != 2:
            raise ValueError('百分位调窗只支持灰度图像')
        values = np.sort(frame[np.isfinite(frame)], axis=None)
        if not values.size:
            raise ValueError('图像中没有可用于统计的有限像素')
        return cls(values, finite_count=int(values.size))

    def window(self, lower, upper):
        if not (np.isfinite(lower) and np.isfinite(upper) and 0 <= lower < upper <= 100):
            raise ValueError('百分位须满足 0 ≤ 下百分位 < 上百分位 ≤ 100')
        count = int(self.counts.sum()) if self.counts is not None else len(self.values)
        ranks = np.array([lower, upper], np.float64)/100*(count-1)
        lo, hi = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
        if self.counts is None:
            a, b = self.values[lo], self.values[hi]
        else:
            cumulative = np.cumsum(self.counts)
            a = self.values[np.searchsorted(cumulative, lo, side='right')]
            b = self.values[np.searchsorted(cumulative, hi, side='right')]
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        low, high = a + (b-a)*(ranks-lo)
        # Constant percentile bounds cannot form a positive display window.
        width = float(high-low) if high > low else 1.
        return float(low), float(high), width, float((low+high)/2)


def collect_statistics(spec, cancelled, progress=None, sample_budget=2_000_000):
    """Exact <=16-bit histograms; bounded stratified samples for larger types."""
    if len(spec.shape) != 2:
        raise ValueError('整组百分位统计只支持灰度图像')
    check_cancelled(cancelled)
    histogram = dtype = None
    samples = []
    offset, finite_count, approximate = 0, 0, False
    per_frame = max(1, sample_budget//len(spec.indices))
    rng = np.random.default_rng(2026)
    with sequence_frames(spec) as frames:
        for position, (number, frame) in enumerate(frames):
            check_cancelled(cancelled)
            if dtype is None:
                dtype = frame.dtype
                if dtype.kind in 'bui' and dtype.itemsize <= 2:
                    info = np.iinfo(dtype) if dtype.kind != 'b' else None
                    offset = -int(info.min) if info is not None else 0
                    bins = int(info.max)-int(info.min)+1 if info is not None else 2
                    histogram = np.zeros(bins, np.int64)
                elif dtype.kind == 'f' and dtype.itemsize == 2:
                    histogram = np.zeros(65536, np.int64)
            if frame.dtype != dtype:
                raise ValueError(f'帧 {number} 的数据类型与首帧不同，未混合统计')
            if histogram is not None:
                if dtype.kind == 'f':
                    indices = frame.astype(np.float16, copy=False).view(np.uint16).ravel()
                else:
                    indices = frame.astype(np.int32).ravel()+offset
                histogram += np.bincount(indices, minlength=len(histogram))
            else:
                flat = frame.ravel()
                finite_count += int(np.count_nonzero(np.isfinite(flat)))
                if flat.size > per_frame:
                    # One sample per equally sized interval avoids periodic
                    # stride aliasing while keeping equal pixel inclusion rates.
                    edges = np.arange(per_frame, dtype=np.float64)*flat.size/per_frame
                    chosen = np.floor(edges+rng.random(per_frame)*flat.size/per_frame).astype(np.int64)
                    sample = flat[chosen]
                    approximate = True
                else:
                    sample = flat
                samples.append(np.array(sample[np.isfinite(sample)], copy=True))
            if progress and (position % 5 == 0 or position+1 == len(spec.indices)):
                progress(f'统计原始灰度 {position+1}/{len(spec.indices)} 帧…')
    check_cancelled(cancelled)
    if histogram is not None:
        if dtype.kind == 'f':
            values = np.arange(65536, dtype=np.uint16).view(np.float16)
        else:
            values = np.arange(len(histogram), dtype=np.int32)-offset
        valid = (histogram > 0) & np.isfinite(values)
        values, counts = values[valid], histogram[valid]
        order = np.argsort(values)
        result = IntensityStatistics(values[order], counts[order], len(spec.indices), int(counts.sum()))
    else:
        values = np.sort(np.concatenate(samples))
        result = IntensityStatistics(values, frames=len(spec.indices), finite_count=finite_count,
                                     approximate=approximate)
    if not len(result.values):
        raise ValueError('整组数据没有可用于统计的有限像素；可尝试增加采样或检查原始数据')
    return result
