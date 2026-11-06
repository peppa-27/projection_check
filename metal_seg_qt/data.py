"""Lazy, read-only image sources. No dependency on training/reconstruction code."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import configparser
from math import isqrt
from pathlib import Path
import re
from threading import Event
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
RAW_EXTENSIONS = {'.prj', '.raw', '.bin', '.slc'}
SUPPORTED = IMAGE_EXTENSIONS | RAW_EXTENSIONS | {'.npy'}


@dataclass(frozen=True)
class ImportOptions:
    domain: str = 'projection'
    mask: bool = False
    height: int = 768
    width: int = 768
    dtype: str = '<u2'
    use_xml: bool = True
    reconstruction_height: int = 800
    reconstruction_width: int = 800
    auto_reconstruction_shape: bool = True


def data_domain(paths, fallback='projection'):
    if any(Path(p).suffix.lower() == '.slc' or
           any(part.lower() in {'slice', 'slices'} for part in Path(p).parent.parts)
           for p in paths):
        return 'reconstruction'
    return fallback


def natural_key(path: Path):
    return [int(p) if p.isdigit() else p.lower()
            for p in re.split(r'(\d+)', path.name)]


def frame_number(path: Path):
    match = re.search(r'(\d+)$', path.stem)
    return int(match[1]) if match else None


def nearby_shape(folder: Path, domain: str):
    """Read this project's XML, including incorrectly declared UTF-16 files."""
    candidates = [folder / 'cbct_task.xml', folder.parent / 'proj' / 'cbct_task.xml',
                  folder.parent / 'cbct_task.xml']
    for path in candidates:
        if not path.is_file():
            continue
        raw = path.read_bytes()
        encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
        text = re.sub(r'<\?xml[^>]*\?>', '', raw.decode(encoding), count=1)
        config = {node.tag: node.get('v') for node in ET.fromstring(text).iter()}
        if domain == 'projection':
            if config.get('Sensor_Vertical_Size') and config.get('Sensor_Horizontal_Size'):
                return int(config['Sensor_Vertical_Size']), int(config['Sensor_Horizontal_Size'])
        elif config.get('Volume_Horizontal_Size'):
            size = int(config['Volume_Horizontal_Size'])
            return size, size
    return None


def reconstruction_shape(paths, options):
    manual = (options.reconstruction_height, options.reconstruction_width)
    if not options.auto_reconstruction_shape:
        return manual, '手动重建尺寸'
    folder = paths[0].parent
    for path in (folder / 'ImageParam.ini', folder.parent / 'ImageParam.ini'):
        if path.is_file():
            raw = path.read_bytes()
            encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
            config = configparser.ConfigParser(interpolation=None, strict=False)
            config.read_string(raw.decode(encoding))
            for section in config.values():
                if 'width' in section and 'height' in section:
                    return (int(section['height']), int(section['width'])), path.name
    if options.use_xml:
        shape = nearby_shape(folder, 'reconstruction')
        if shape:
            return shape, 'XML 重建尺寸'
    if paths[0].suffix.lower() == '.slc':
        size = paths[0].stat().st_size
        itemsize = np.dtype('<i2').itemsize  # SLC always stores signed 16-bit slices.
        side = isqrt(size // itemsize)
        if side > 0 and side * side * itemsize == size:
            return (side, side), '按 SLC 字节数推导（方形切片）'
        raise ValueError('无法推导 SLC 尺寸；请取消重建尺寸自动识别并填写实际宽、高和 dtype')
    return manual, '手动重建尺寸（无元数据）'


def read_image(path: Path, page=0):
    with Image.open(path) as image:
        image.seek(page)
        if image.mode == 'P' or image.mode in ('CMYK', 'YCbCr', 'LA'):
            image = image.convert('RGBA' if 'transparency' in image.info or image.mode == 'LA' else 'RGB')
        return np.array(image)


def validate_frame(frame):
    if frame.ndim != 2 and not (frame.ndim == 3 and frame.shape[-1] in (3, 4)):
        raise ValueError(f'不支持的图像形状：{frame.shape}')
    if not frame.size or any(n == 0 for n in frame.shape):
        raise ValueError('图像不能为空')
    if frame.dtype.kind not in 'buif':
        raise ValueError(f'不支持的数据类型：{frame.dtype}')


class Dataset:
    """A sequence indexed by actual filename numbers, or by zero-based array index."""
    def __init__(self, paths, options: ImportOptions):
        self.id = uuid.uuid4().hex
        self._temporary_storage = None
        self._section_cache = None
        self.generated = False
        self.description = ''
        self.paths = tuple(sorted((Path(p) for p in paths), key=natural_key))
        self.geometry_paths = self.paths
        if not self.paths:
            raise ValueError('没有可读取的图像')
        # The project's slice/SLC raw format is signed 16-bit little endian.
        # Keep this independent of projection/other-raw import settings, also
        # when both domains are imported together from a sample directory.
        if data_domain(self.paths) == 'reconstruction':
            options = replace(options, dtype='<i2')
        self.domain, self.mask = data_domain(self.paths, options.domain), options.mask
        self.name = self.paths[0].parent.name if len(self.paths) > 1 else self.paths[0].name
        self._array = None
        self._cache = OrderedDict()
        self._raw_shape = (options.height, options.width)
        self._raw_dtype = np.dtype(options.dtype)
        self.shape_source = '图像 / 数组自带尺寸'
        self._pages = False
        suffix = self.paths[0].suffix.lower()
        self.kind = 'image'
        count = len(self.paths)
        if suffix == '.npy':
            self.kind = 'array'
            self._array = np.load(self.paths[0], mmap_mode='r', allow_pickle=False)
            # Arrays are deliberately unambiguous: grayscale [H,W] or [N,H,W].
            if self._array.ndim not in (2, 3) or self._array.dtype.kind not in 'buif':
                self.close()
                raise ValueError('NPY 必须是数值灰度数组 [H,W] 或 [N,H,W]')
            count = 1 if self._array.ndim == 2 else self._array.shape[0]
        elif suffix in RAW_EXTENSIONS:
            self.kind = 'raw'
            if self.domain == 'reconstruction':
                self._raw_shape, self.shape_source = reconstruction_shape(self.paths, options)
            elif options.use_xml:
                xml_shape = nearby_shape(self.paths[0].parent, self.domain)
                self._raw_shape = xml_shape or self._raw_shape
                self.shape_source = 'XML 投影尺寸' if xml_shape else '手动投影尺寸'
            else:
                self.shape_source = '手动投影尺寸'
            if min(self._raw_shape) <= 0:
                raise ValueError('原始图像的宽、高必须大于零')
            frame_bytes = int(np.prod(self._raw_shape)) * self._raw_dtype.itemsize
            if len(self.paths) == 1 and suffix in {'.bin', '.raw'}:
                size = self.paths[0].stat().st_size
                if size == 0 or size % frame_bytes:
                    raise ValueError('原始体数据字节数与宽、高、dtype 不匹配，请调整导入参数')
                count = size // frame_bytes
                self._array = np.memmap(self.paths[0], dtype=self._raw_dtype, mode='r',
                                        shape=(count, *self._raw_shape))
            else:
                for path in self.paths:
                    if path.stat().st_size != frame_bytes:
                        raise ValueError(f'{path.name} 大小与 {self._raw_shape} / {self._raw_dtype} 不匹配')
        elif suffix in {'.tif', '.tiff'} and len(self.paths) == 1:
            with Image.open(self.paths[0]) as image:
                count = image.n_frames
            self._pages = count > 1
        if count < 1:
            self.close()
            raise ValueError('序列不能为空')
        numbered_files = self._array is None and not self._pages
        numbers = [frame_number(p) for p in self.paths] if numbered_files else []
        self.numbered = bool(numbers) and all(n is not None for n in numbers)
        self.indices = tuple(numbers) if self.numbered else tuple(range(count))
        if len(set(self.indices)) != len(self.indices):
            raise ValueError('存在重复层号，请将不同处理结果放在不同文件夹中')
        if max(self.indices) > 2_000_000_000:
            raise ValueError('文件名末尾数字过大，不能用作层号；请使用 proj_0 这样的名称')
        self._positions = {number: i for i, number in enumerate(self.indices)}
        try:
            frame = self.frame(self.indices[0])
            self.shape, self.dtype = frame.shape, str(frame.dtype)
        except Exception:
            self.close()
            raise

    def frame(self, number):
        if number not in self._positions:
            raise KeyError(number)
        if number in self._cache:
            self._cache.move_to_end(number)
            return self._cache[number]
        position = self._positions[number]
        if self._array is not None:
            frame = np.array(self._array if self._array.ndim == 2 else self._array[position])
        elif self.kind == 'raw':
            frame = np.fromfile(self.paths[position], dtype=self._raw_dtype).reshape(self._raw_shape)
        else:
            frame = read_image(self.paths[0] if self._pages else self.paths[position],
                               position if self._pages else 0)
        validate_frame(frame)
        if hasattr(self, 'shape') and frame.shape != self.shape:
            raise ValueError(f'层 {number} 的尺寸 {frame.shape} 与首层 {self.shape} 不一致')
        self._cache[number] = frame
        while len(self._cache) > 4:
            self._cache.popitem(last=False)
        return frame

    def close(self):
        self._cache.clear()
        mmap = getattr(self._array, '_mmap', None)
        if mmap is not None:
            mmap.close()
        self._array = None
        # Background SequenceSpecs retain storage until their own readers close.
        self._temporary_storage = None
        self._section_cache = None

    def __del__(self):
        # Also close mappings before deleting temporary storage when a result
        # is released after its window/event loop has already shut down.
        if hasattr(self, '_cache'):
            self.close()


def discover_groups(paths):
    """Load this folder and each direct child as separate image sequences."""
    groups = []
    loose = {}

    def add_folder(folder):
        files = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED),
                       key=natural_key)
        if not files:
            return False
        grouped = {}
        for path in files:
            if path.suffix.lower() == '.npy' or path.suffix.lower() in {'.bin', '.raw'}:
                groups.append([path])
            else:
                key = 'image' if path.suffix.lower() in IMAGE_EXTENSIONS else path.suffix.lower()
                grouped.setdefault(key, []).append(path)
        groups.extend(grouped.values())
        return True

    for value in paths:
        path = Path(value).resolve()
        if path.is_dir():
            found = add_folder(path)
            for child in sorted(path.iterdir(), key=natural_key):
                if child.is_dir():
                    found = add_folder(child) or found
            if not found:
                raise ValueError(f'{path.name} 中没有支持的图像（只扫描当前目录及下一层）')
        elif path.suffix.lower() in SUPPORTED:
            # A multi-file drop from the same folder forms one sequence.
            suffix = path.suffix.lower()
            key = (path.parent, 'image' if suffix in IMAGE_EXTENSIONS else suffix)
            if suffix in {'.npy', '.bin', '.raw'}:
                groups.append([path])
            else:
                loose.setdefault(key, []).append(path)
        else:
            raise ValueError(f'不支持的文件：{path.name}')
    groups.extend(loose.values())
    unique = {}
    for group in groups:
        key = tuple(sorted(set(group), key=natural_key))
        unique[key] = list(key)
    return list(unique.values())


def load_datasets(paths, options, cancelled: Event | None = None):
    datasets, errors = [], []
    try:
        groups = discover_groups(paths)
        for group in groups:
            if cancelled is not None and cancelled.is_set():
                break
            try:
                datasets.append(Dataset(group, options))
            except Exception as error:
                errors.append(f'{group[0].parent.name}/{group[0].name}：{error}')
    except Exception as error:
        errors.append(str(error))
    if cancelled is not None and cancelled.is_set():
        for dataset in datasets:
            dataset.close()
        return [], []
    return datasets, errors


def auto_window(frame, selection=None):
    if selection is not None:
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = np.clip(selection, 0., 1.)
        left = min(width - 1, int(np.floor(min(x0, x1) * width)))
        top = min(height - 1, int(np.floor(min(y0, y1) * height)))
        right = min(width, max(left + 1, int(np.ceil(max(x0, x1) * width))))
        bottom = min(height, max(top + 1, int(np.ceil(max(y0, y1) * height))))
        frame = frame[top:bottom, left:right]
    sample = frame.ravel()[::max(1, frame.size // 200_000)]
    finite = sample[np.isfinite(sample)]
    if not finite.size:
        return 1., .5
    low, high = (float(v) for v in np.percentile(finite, [1, 99]))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    # Keep zero masks black and nonzero constant masks visible.
    if high <= low:
        return max(abs(low), 1.), max(abs(low), 1.) / 2
    return high - low, (high + low) / 2


def display_pixels(frame, width, level, mask=False):
    if mask:
        if frame.ndim == 3:
            frame = frame[..., :3].max(axis=-1)
        return (np.isfinite(frame) & (frame != 0)).astype(np.uint8) * 255
    if frame.ndim == 3:
        return np.ascontiguousarray(np.clip(frame, 0, 255), dtype=np.uint8)
    if not np.isfinite(width) or width <= 0 or not np.isfinite(level):
        raise ValueError('窗宽必须为有限正数，窗位必须为有限数')
    values = frame.astype(np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        values = (values - (level - width / 2)) / width
    values = np.nan_to_num(values, nan=0., posinf=1., neginf=0.)
    return np.ascontiguousarray(np.rint(np.clip(values, 0, 1) * 255), dtype=np.uint8)
