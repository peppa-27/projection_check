"""Calibrated voxel -> detector point, and detector point -> clipped 3-D ray."""
from dataclasses import dataclass
from pathlib import Path
import configparser
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np


class GeometryError(ValueError):
    pass


@dataclass
class ReconstructionRay:
    local_endpoints: np.ndarray
    world_endpoints: np.ndarray
    slice_begin: int
    slice_shape: tuple

    def normalized_xy(self, points):
        height, width = self.slice_shape
        return tuple(tuple(float(v) for v in ((p[:2] + .5) / [width, height])) for p in points)

    @property
    def top_view(self):
        """2-D projection of the 3-D path, not its intersection with a slice."""
        return self.normalized_xy(self.local_endpoints)

    def section(self, layer):
        """Return (point, coplanar_segment); neither means no plane intersection."""
        a, b = self.local_endpoints
        depth = layer - self.slice_begin
        delta = b - a
        if abs(delta[2]) <= 1e-9:
            if abs(a[2] - depth) <= 1e-8:
                return None, self.top_view
            return None, None
        t = (depth - a[2]) / delta[2]
        if not -1e-9 <= t <= 1 + 1e-9:
            return None, None
        point = a + np.clip(t, 0, 1) * delta
        return self.normalized_xy([point])[0], None


def case_root(dataset):
    folder = getattr(dataset, 'geometry_paths', dataset.paths)[0].resolve().parent
    candidates = [folder, *list(folder.parents)[:2]]
    # Prefer the original case/proj geometry to copies in processed folders.
    for root in candidates:
        if (root / 'proj' / 'PTM.dat').is_file() and (root / 'proj' / 'cbct_task.xml').is_file():
            return root
    for root in candidates:
        if (root / 'PTM.dat').is_file() and (root / 'cbct_task.xml').is_file():
            return root.parent if root.name.lower() == 'proj' else root
    raise GeometryError('找不到当前样本的 proj/PTM.dat 和 cbct_task.xml')


def read_xml(path):
    raw = path.read_bytes()
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    text = re.sub(r'<\?xml[^>]*\?>', '', raw.decode(encoding), count=1)
    return {node.tag: node.get('v') for node in ET.fromstring(text).iter() if node.get('v') is not None}


def read_ini(path):
    raw = path.read_bytes()
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read_string(raw.decode(encoding))
    # Only geometric attributes are retained; patient information is unused.
    required = {'width', 'height', 'beginindex', 'endindex', 'manufacturer',
                *(prefix + axis for prefix in ('imageposition_', 'cosrow_', 'coscolumn_', 'scale_')
                  for axis in 'xyz')}
    return {key: value for section in parser.values() for key, value in section.items() if key in required}


@dataclass
class ProjectionGeometry:
    root: Path
    matrices: np.ndarray
    detector_shape: tuple
    projection_start: int
    slice_begin: int
    slice_end: int
    slice_shape: tuple
    origin: np.ndarray
    column_step: np.ndarray
    row_step: np.ndarray
    slice_step: np.ndarray
    ini_path: Path
    coordinate_convention: str = 'INI 原坐标（未指定厂家转换）'

    @classmethod
    def from_dataset(cls, dataset):
        override = getattr(dataset, 'geometry_override', None)
        if override is not None:
            if (tuple(dataset.shape[:2]) != override.slice_shape or
                    min(dataset.indices) < override.slice_begin or max(dataset.indices) > override.slice_end):
                raise GeometryError('暂存体数据与其物理网格不一致')
            return override
        try:
            root = case_root(dataset)
            folder = root / 'proj' if (root / 'proj' / 'PTM.dat').is_file() else root
            config = read_xml(folder / 'cbct_task.xml')
            values = np.loadtxt(folder / 'PTM.dat', dtype=np.float64, ndmin=2)
            if values.shape[1] != 12 or not np.isfinite(values).all():
                raise GeometryError('PTM 每行必须是 12 个有限数值（3×4 矩阵）')
            matrices = values.reshape(-1, 3, 4)
            if any(np.linalg.matrix_rank(matrix) != 3 for matrix in matrices):
                raise GeometryError('PTM 中存在退化投影矩阵')
            if int(config['Input_Views_Number']) != len(matrices):
                raise GeometryError('PTM 行数与 XML 投影数量不一致')
            detector = (int(config['Sensor_Vertical_Size']), int(config['Sensor_Horizontal_Size']))
            if min(detector) <= 0:
                raise GeometryError('探测器尺寸无效')
            source = getattr(dataset, 'geometry_paths', dataset.paths)[0].resolve().parent
            ini_path = next((p for p in (source / 'ImageParam.ini', root / 'slice' / 'ImageParam.ini')
                             if p.is_file()), None)
            if ini_path is None:
                raise GeometryError('缺少重建 ImageParam.ini，无法确定三维位置、方向和层序')
            ini = read_ini(ini_path)
            shape = (int(ini['height']), int(ini['width']))
            if tuple(dataset.shape[:2]) != shape:
                raise GeometryError('重建图像尺寸与几何元数据不一致；暂不推测裁剪或重采样关系')
            begin, end = int(ini['beginindex']), int(ini['endindex'])
            if end < begin:
                raise GeometryError('重建层号范围无效')
            if min(dataset.indices) < begin or max(dataset.indices) > end:
                raise GeometryError('重建层号超出 ImageParam.ini 的 BeginIndex / EndIndex')
            vector = lambda prefix: np.array([float(ini[prefix + axis]) for axis in 'xyz'])
            origin, row, column, spacing = (vector(p) for p in
                                           ('imageposition_', 'cosrow_', 'coscolumn_', 'scale_'))
            if not np.isfinite([origin, row, column, spacing]).all() or np.any(spacing <= 0):
                raise GeometryError('重建坐标或体素间距无效')
            if not (np.isclose(np.linalg.norm(row), 1, atol=1e-3) and
                    np.isclose(np.linalg.norm(column), 1, atol=1e-3) and
                    np.isclose(row @ column, 0, atol=1e-3)):
                raise GeometryError('重建行列方向不是正交单位向量')
            if any(int(config.get('Volume_Mirroring_' + axis, 0)) != 0 for axis in 'XYZ'):
                raise GeometryError('当前几何定位暂不支持 XML 中额外的体数据镜像变换')
            normal = np.cross(row, column)
            # Fussen native SLC INIs use image/patient axes, while these PTMs
            # use the scanner frame. X and Z point in opposite directions.
            # This is a coordinate transform, NOT a display/image-array flip.
            # Verified with native metal landmarks across projection angles;
            # do not apply the vendor-specific convention to generic inputs.
            frame = np.eye(3)
            convention = 'INI 原坐标（未指定厂家转换）'
            if ini.get('manufacturer', '').strip().casefold() == 'fussen':
                frame = np.diag([-1., 1., -1.])
                convention = 'Fussen 原生 INI → PTM：(-X, +Y, -Z)'
            ptm_center = np.array([float(config.get('Volume_Translation_' + axis, 0)) for axis in 'XYZ'])
            center = frame.T @ ptm_center
            # The INI locates the FIRST voxel centre, rather than a volume
            # boundary. Infer signed storage order from that plane and the
            # XML volume centre; the vendor's native slices usually descend Z.
            offset = float((center - origin) @ normal)
            half_depth = (end - begin) * spacing[2] / 2
            if not np.isfinite(center).all() or abs(abs(offset) - half_depth) > max(spacing[2], 1e-3):
                raise GeometryError('首层位置与 XML 体积中心不一致，无法可靠确定重建层序')
            direction = -1. if offset < 0 else 1.
            return cls(root, matrices, detector, int(config.get('Input_Views_Start_Count', 0)),
                       begin, end, shape, frame @ origin, frame @ (row * spacing[0]),
                       frame @ (column * spacing[1]), frame @ (direction * normal * spacing[2]),
                       ini_path, convention)
        except GeometryError:
            raise
        except Exception as error:
            raise GeometryError(f'几何文件读取失败：{error}') from error

    @classmethod
    def from_projection_dataset(cls, dataset):
        """Obtain the calibrated reconstruction grid without requiring loaded SLCs."""
        root = case_root(dataset)
        paths = getattr(dataset, 'geometry_paths', dataset.paths)
        # Projection folders can also contain ImageParam.ini with detector
        # attributes; the reconstruction grid must come from slice metadata.
        candidates = (root / 'slice' / 'ImageParam.ini', paths[0].parent / 'ImageParam.ini')
        ini_path = next((p for p in candidates if p.is_file()), None)
        if ini_path is None:
            raise GeometryError('缺少 ImageParam.ini，无法确定重建的物理网格')
        try:
            values = read_ini(ini_path)
            geometry_paths = (ini_path.parent / 'slice_0.slc',)
            descriptor = SimpleNamespace(paths=geometry_paths, geometry_paths=geometry_paths,
                shape=(int(values['height']), int(values['width'])),
                indices=(int(values['beginindex']), int(values['endindex'])))
            return cls.from_dataset(descriptor)
        except (KeyError, ValueError) as error:
            raise GeometryError(f'重建网格读取失败：{error}') from error

    def voxel(self, normalized, layer):
        if not self.slice_begin <= layer <= self.slice_end:
            raise GeometryError('当前重建层号不在几何范围内')
        height, width = self.slice_shape
        x, y = normalized
        if not np.isfinite([x, y]).all() or not 0 <= x <= 1 or not 0 <= y <= 1:
            raise GeometryError('选点不在重建图像内')
        col = min(width - 1, int(np.floor(x * width)))
        row = min(height - 1, int(np.floor(y * height)))
        world = (self.origin + col * self.column_step + row * self.row_step +
                 (layer - self.slice_begin) * self.slice_step)
        return (col, row, layer), world

    def project(self, world, projection_number, image_shape):
        index = projection_number - self.projection_start
        if not 0 <= index < len(self.matrices):
            raise GeometryError(f'投影层号 {projection_number} 没有对应的 PTM 行')
        height, width = self.detector_shape
        shown_height, shown_width = image_shape[:2]
        if not np.isclose(shown_width / shown_height, width / height, rtol=1e-3):
            raise GeometryError('显示图像比例与探测器不符，不能直接映射（可能经过裁剪或拼接）')
        point = self.matrices[index] @ np.r_[world, 1.]
        if not np.isfinite(point).all() or point[2] <= 1e-12:
            raise GeometryError('选点位于当前探测器背侧或投影无效')
        u, v = point[:2] / point[2]
        # PTM integers denote pixel centres, while QPixmap coordinates start
        # at the first pixel's outer edge. Preserve the half-pixel conversion.
        if not (-.5 <= u < width - .5 and -.5 <= v < height - .5):
            raise GeometryError('选点投影位于当前探测器视野之外')
        return (float((u + .5) / width), float((v + .5) / height))

    def check_projection_dataset(self, dataset):
        if case_root(dataset) != self.root:
            raise GeometryError('投影与重建不属于同一个几何目录，未绘制对应点')

    def backproject(self, normalized, projection_number, image_shape):
        index = projection_number - self.projection_start
        if not 0 <= index < len(self.matrices):
            raise GeometryError(f'投影层号 {projection_number} 没有对应的 PTM 行')
        height, width = self.detector_shape
        if not np.isclose(image_shape[1] / image_shape[0], width / height, rtol=1e-3):
            raise GeometryError('显示图像比例与探测器不符，不能反投影')
        x, y = normalized
        if not np.isfinite([x, y]).all() or not (0 <= x < 1 and 0 <= y < 1):
            raise GeometryError('选点不在投影图像内')
        matrix = self.matrices[index]
        try:
            # P = [M|p], source C = -M^-1 p. The positive half-ray
            # C + t M^-1[u,v,1] projects to this pixel for every t > 0.
            source = np.linalg.solve(matrix[:, :3], -matrix[:, 3])
            direction = np.linalg.solve(matrix[:, :3], [x * width - .5, y * height - .5, 1.])
            direction /= np.linalg.norm(direction)
            basis = np.column_stack((self.column_step, self.row_step, self.slice_step))
            local_source = np.linalg.solve(basis, source - self.origin)
            local_direction = np.linalg.solve(basis, direction)
        except np.linalg.LinAlgError as error:
            raise GeometryError('投影矩阵无法还原有限焦点射线') from error
        if not np.isfinite([source, direction, local_source, local_direction]).all():
            raise GeometryError('反投影射线包含无效数值')
        counts = np.array([self.slice_shape[1], self.slice_shape[0], self.slice_end - self.slice_begin + 1])
        lo, hi = np.full(3, -.5), counts - .5
        enter, leave = 0., np.inf
        for axis in range(3):
            if abs(local_direction[axis]) <= 1e-12:
                if not lo[axis] <= local_source[axis] <= hi[axis]:
                    raise GeometryError('所选射线不穿过当前重建体积')
                continue
            bounds = (np.array([lo[axis], hi[axis]]) - local_source[axis]) / local_direction[axis]
            enter, leave = max(enter, bounds.min()), min(leave, bounds.max())
            if enter > leave:
                raise GeometryError('所选射线不穿过当前重建体积')
        if not np.isfinite([enter, leave]).all():
            raise GeometryError('无法确定射线与重建体积的交段')
        distances = np.array([enter, leave])[:, None]
        local = np.clip(local_source + distances * local_direction, lo, hi)
        world = source + distances * direction
        return ReconstructionRay(local, world, self.slice_begin, self.slice_shape)
