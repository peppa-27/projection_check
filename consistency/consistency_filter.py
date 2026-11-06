"""PTM-based implementation of Gottschalk et al., 2021, Sec. 2.2.2.

No network or ground truth is required. Arrays use (view, row, column) and
(z, y, x); PTM maps world (x, y, z, 1) to homogeneous detector (u, v, 1).
The binary volume is a silhouette envelope, NOT an attenuation reconstruction.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import re
import xml.etree.ElementTree as ET

import numpy as np
from numba import njit, prange


@dataclass(frozen=True)
class VolumeGrid:
    shape_zyx: tuple[int, int, int]
    spacing_xyz: tuple[float, float, float]
    # Coordinates of the LOWER voxel boundary, not the first voxel center.
    lower_xyz: tuple[float, float, float]

    def __post_init__(self):
        if len(self.shape_zyx) != 3 or any(n <= 0 for n in self.shape_zyx):
            raise ValueError('Grid shape must contain three positive integers')
        if len(self.spacing_xyz) != 3 or any(s <= 0 for s in self.spacing_xyz):
            raise ValueError('Grid spacing must be positive')
        if not np.isfinite((*self.spacing_xyz, *self.lower_xyz)).all():
            raise ValueError('Grid coordinates must be finite')


def read_xml_config(path):
    """Accept the supplied XML, whose UTF-16 declaration has UTF-8 bytes."""
    raw = Path(path).read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        text = raw.decode('utf-16')
    else:
        text = raw.decode('utf-8-sig')
    text = re.sub(r'<\?xml[^>]*\?>', '', text, count=1)
    root = ET.fromstring(text)
    return {node.tag: node.get('v') for node in root if node.get('v') is not None}


def grid_from_config(config, downsample=1):
    """Centered world volume; preserve physical extent when reducing resolution."""
    if not isinstance(downsample, int) or downsample < 1:
        raise ValueError('downsample must be a positive integer')
    nx = ny = int(config['Volume_Horizontal_Size'])
    nz = int(config['Volume_Vertical_Size'])
    native = np.array([nx, ny, nz])
    spacing = np.array([float(config['Volume_Horizontal_Element_Size'])] * 2 +
                       [float(config['Volume_Vertical_Element_Size'])])
    extent = native * spacing
    center = np.array([float(config.get('Volume_Translation_' + a, 0)) for a in 'XYZ'])
    shape = np.maximum(1, (native + downsample - 1) // downsample)
    return VolumeGrid(tuple(int(n) for n in shape[::-1]),
                      tuple(extent / shape), tuple(center - extent / 2))


def load_ptm(path):
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if a.shape[1] != 12 or not np.isfinite(a).all():
        raise ValueError('PTM must contain 12 finite numbers per view')
    matrices = np.ascontiguousarray(a.reshape(-1, 3, 4))
    if np.any(np.abs(np.linalg.det(matrices[:, :, :3])) < 1e-12):
        raise ValueError('Singular PTM matrix')
    return matrices


def indexed_prj_files(folder):
    files = {}
    for path in Path(folder).iterdir():
        match = re.fullmatch(r'proj_(\d+)\.prj', path.name, flags=re.I)
        if match:
            index = int(match.group(1))
            if index in files:
                raise ValueError(f'Duplicate projection index: {index}')
            files[index] = path
    if not files:
        raise ValueError(f'No proj_<index>.prj files in {folder}')
    return dict(sorted(files.items()))


def write_binary_mask(path, mask):
    """Write detector masks as little-endian uint16 with integer values 0/1.

    Never preserve an input float16 encoding: its 1.0 bit pattern is 15360
    when read by the project's uint16 PRJ consumers.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2 or not np.isin(mask, [0, 1]).all():
        raise ValueError('Output mask must be a 2D array containing only 0 and 1')
    mask.astype('<u2').tofile(path)


def read_binary_mask(path, shape, dtype='auto', threshold=0.5):
    """Auto accepts ONLY unambiguous binary encodings, never arbitrary intensities."""
    path = Path(path)
    pixels = int(np.prod(shape))
    if dtype != 'auto':
        a = np.fromfile(path, dtype=np.dtype(dtype))
        if a.size != pixels or not np.isfinite(a).all():
            raise ValueError(f'Invalid mask size or values: {path}')
        return (a.reshape(shape) > threshold)
    size = path.stat().st_size
    if size == pixels:
        a = np.fromfile(path, dtype=np.uint8)
        if np.isin(a, [0, 1, 255]).all():
            return (a != 0).reshape(shape)
    elif size == pixels * 2:
        a = np.fromfile(path, dtype='<u2')
        if np.isin(a, [0, 1, 255, 65535]).all():
            return (a != 0).reshape(shape)
        f = a.view('<f2')
        if np.isin(f, [0., 1.]).all():
            return (f > threshold).reshape(shape)
    elif size == pixels * 4:
        a = np.fromfile(path, dtype='<f4')
        if np.isin(a, [0., 1.]).all():
            return (a > threshold).reshape(shape)
    raise ValueError(f'Ambiguous mask encoding in {path}; specify dtype explicitly')


@njit(cache=True, parallel=True)
def _backproject_chunk(masks, matrices, order, lower, spacing, nx, ny,
                       z_start, z_count, threshold, min_valid_views):
    """Paper Eq. 3-5. Early rejection cannot discard a passing voxel."""
    n, height, width = masks.shape
    result = np.zeros((z_count, ny, nx), dtype=np.bool_)
    # If misses exceed this, even all N remaining/valid views cannot reach tau.
    max_misses = int(np.floor((1.0 - threshold) * n + 1e-9))
    for iz in prange(z_count):
        z = lower[2] + (z_start + iz + 0.5) * spacing[2]
        for iy in range(ny):
            y = lower[1] + (iy + 0.5) * spacing[1]
            for ix in range(nx):
                x = lower[0] + (ix + 0.5) * spacing[0]
                hits = 0
                valid = 0
                rejected = False
                for oi in range(n):
                    i = order[oi]
                    p = matrices[i]
                    w = p[2, 0]*x + p[2, 1]*y + p[2, 2]*z + p[2, 3]
                    if w <= 1e-12:
                        continue
                    u = (p[0, 0]*x + p[0, 1]*y + p[0, 2]*z + p[0, 3]) / w
                    v = (p[1, 0]*x + p[1, 1]*y + p[1, 2]*z + p[1, 3]) / w
                    # Integer coordinates denote pixel centers (nearest neighbor).
                    if -0.5 <= u < width - 0.5 and -0.5 <= v < height - 0.5:
                        valid += 1
                        if masks[i, int(np.floor(v+0.5)), int(np.floor(u+0.5))]:
                            hits += 1
                        if valid - hits > max_misses:
                            rejected = True
                            break
                if not rejected and valid >= min_valid_views:
                    result[iz, iy, ix] = hits + 1e-9 >= threshold * valid
    return result


def backproject_consensus(masks, matrices, grid, threshold=0.96,
                          min_valid_views=1, chunk_depth=16, progress=None):
    """Return binary 3-D consensus envelope; out-of-detector views are excluded.

    min_valid_views=1 follows the paper. A higher value is an optional extra
    coverage safeguard, not part of the reported paper default.
    """
    masks = np.ascontiguousarray(masks, dtype=np.bool_)
    matrices = np.ascontiguousarray(matrices, dtype=np.float64)
    if masks.ndim != 3 or matrices.shape != (len(masks), 3, 4):
        raise ValueError('Expected masks [N,H,W] and matching PTM [N,3,4]')
    if not np.isfinite(matrices).all() or not 0 < threshold <= 1:
        raise ValueError('Invalid PTM or consistency threshold')
    if not 1 <= min_valid_views <= len(masks) or chunk_depth < 1:
        raise ValueError('Invalid minimum view count or chunk depth')
    # Spread early samples over the trajectory; final decision is order invariant.
    order = np.random.default_rng(0).permutation(len(masks))
    nz, ny, nx = grid.shape_zyx
    volume = np.zeros(grid.shape_zyx, dtype=np.bool_)
    for start in range(0, nz, chunk_depth):
        count = min(chunk_depth, nz-start)
        volume[start:start+count] = _backproject_chunk(
            masks, matrices, order, np.array(grid.lower_xyz),
            np.array(grid.spacing_xyz), nx, ny, start, count,
            threshold, min_valid_views)
        if progress:
            progress(f'Backprojection: {start+count}/{nz} z planes')
    return volume


@njit(cache=True)
def _ray_hits(volume, source, direction, lower, spacing):
    """Exact voxel traversal (DDA): true iff positive-length metal path exists."""
    sizes = (volume.shape[2], volume.shape[1], volume.shape[0])
    t_enter = 0.0
    t_exit = np.inf
    for a in range(3):
        hi = lower[a] + sizes[a]*spacing[a]
        if abs(direction[a]) < 1e-14:
            if source[a] < lower[a] or source[a] >= hi:
                return False
        else:
            t0 = (lower[a]-source[a])/direction[a]
            t1 = (hi-source[a])/direction[a]
            t_enter = max(t_enter, min(t0, t1))
            t_exit = min(t_exit, max(t0, t1))
    if t_exit <= t_enter:
        return False
    index = np.empty(3, dtype=np.int64)
    step = np.empty(3, dtype=np.int64)
    t_next = np.empty(3, dtype=np.float64)
    t_delta = np.empty(3, dtype=np.float64)
    for a in range(3):
        coord = (source[a]+t_enter*direction[a]-lower[a])/spacing[a]
        index[a] = min(sizes[a]-1, max(0, int(np.floor(coord))))
        if direction[a] > 1e-14:
            step[a] = 1
            boundary = lower[a]+(index[a]+1)*spacing[a]
            t_next[a] = (boundary-source[a])/direction[a]
            t_delta[a] = spacing[a]/direction[a]
        elif direction[a] < -1e-14:
            step[a] = -1
            boundary = lower[a]+index[a]*spacing[a]
            t_next[a] = (boundary-source[a])/direction[a]
            t_delta[a] = -spacing[a]/direction[a]
        else:
            step[a] = 0
            t_next[a] = np.inf
            t_delta[a] = np.inf
    t = t_enter
    while t < t_exit:
        next_t = min(t_next[0], t_next[1], t_next[2], t_exit)
        if next_t > t + 1e-12 and volume[index[2], index[1], index[0]]:
            return True
        if next_t >= t_exit:
            break
        # Step all crossed axes, so corner-only contacts do not count as metal.
        for a in range(3):
            if t_next[a] <= next_t + 1e-12:
                index[a] += step[a]
                if index[a] < 0 or index[a] >= sizes[a]:
                    return False
                t_next[a] += t_delta[a]
        t = next_t
    return False


@njit(cache=True, parallel=True)
def _forward(volume, inverse, sources, bounds, lower, spacing, height, width):
    out = np.zeros((len(inverse), height, width), dtype=np.bool_)
    # Independent detector rows provide enough parallel work even for few views.
    for task in prange(len(inverse)*height):
        i = task // height
        v = task % height
        if v < bounds[i, 2] or v >= bounds[i, 3]:
            continue
        direction = np.empty(3, dtype=np.float64)
        for u in range(bounds[i, 0], bounds[i, 1]):
            for a in range(3):
                direction[a] = inverse[i, a, 0]*u + inverse[i, a, 1]*v + inverse[i, a, 2]
            out[i, v, u] = _ray_hits(volume, sources[i], direction, lower, spacing)
    return out


def forward_project_envelope(volume, matrices, grid, detector_shape, progress=None,
                             *, out=None, batch_size=30):
    """Binary silhouette = (metal path length > 0), with exact voxel traversal.

    Unlike projecting voxel centers/splatting, ray tracing does not leave holes
    between magnified voxels. No intersection with the input masks is applied.
    Optional out receives binary results in place (including disk-backed uint8
    arrays); batch_size controls progress/cancellation callback granularity.
    """
    volume = np.asarray(volume, dtype=np.bool_)
    matrices = np.asarray(matrices, dtype=np.float64)
    if volume.shape != grid.shape_zyx:
        raise ValueError('Volume shape does not match grid')
    height, width = detector_shape
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    shape = (len(matrices), height, width)
    if out is None:
        out = np.zeros(shape, dtype=np.bool_)
    elif out.shape != shape or out.dtype not in (np.dtype('bool'), np.dtype('uint8')):
        raise ValueError('Output must be a bool/uint8 array with shape [N,H,W]')
    active_z = np.flatnonzero(volume.any(axis=(1, 2)))
    if not len(active_z):
        if progress:
            progress('WARNING: consensus volume is empty; check geometry and input masks')
        out[...] = 0
        return out
    active_y = np.flatnonzero(volume.any(axis=(0, 2)))
    active_x = np.flatnonzero(volume.any(axis=(0, 1)))
    lo = np.array([active_x[0], active_y[0], active_z[0]])
    hi = np.array([active_x[-1]+1, active_y[-1]+1, active_z[-1]+1])
    spacing = np.array(grid.spacing_xyz)
    lower = np.array(grid.lower_xyz) + lo*spacing
    upper = np.array(grid.lower_xyz) + hi*spacing
    cropped = np.ascontiguousarray(volume[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]])
    inverse = np.linalg.inv(matrices[:, :, :3])
    sources = -np.einsum('nij,nj->ni', inverse, matrices[:, :, 3])
    corners = np.array([[x,y,z,1.] for x in [lower[0],upper[0]]
                        for y in [lower[1],upper[1]] for z in [lower[2],upper[2]]])
    projected = np.einsum('nij,kj->nki', matrices, corners)
    bounds = np.tile([0,width,0,height], (len(matrices),1)).astype(np.int64)
    for i, points in enumerate(projected):
        if np.all(points[:,2] > 1e-12):
            uv = points[:,:2] / points[:,2:3]
            bounds[i] = [np.clip(np.floor(uv[:,0].min())-1,0,width),
                         np.clip(np.ceil(uv[:,0].max())+2,0,width),
                         np.clip(np.floor(uv[:,1].min())-1,0,height),
                         np.clip(np.ceil(uv[:,1].max())+2,0,height)]
    if progress:
        progress(f'Forward projection: occupied grid bounds xyz {lo.tolist()}..{hi.tolist()}')
    # Batch to bound temporary memory and expose progress.
    for start in range(0,len(matrices),batch_size):
        end = min(start+batch_size,len(matrices))
        out[start:end] = _forward(cropped,inverse[start:end],sources[start:end],
                                  bounds[start:end],lower,spacing,height,width)
        if progress:
            progress(f'Forward projection: {end}/{len(matrices)} views')
    return out


def consistency_filter(masks, matrices, grid, threshold=0.96,
                       min_valid_views=1, progress: Callable[[str], None] | None=None):
    """Embedding API: return (filtered_masks, intermediate_binary_volume)."""
    volume = backproject_consensus(masks,matrices,grid,threshold,min_valid_views,
                                   progress=progress)
    filtered = forward_project_envelope(volume,matrices,grid,masks.shape[1:],progress)
    return filtered, volume
