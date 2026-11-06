"""Display-only mask overlays; source pixels and image geometry stay unchanged."""
import numpy as np


def foreground(frame):
    return np.isfinite(frame) & (frame != 0)


def mask_boundary(mask):
    """One-pixel inner boundary, including image edges, without wrapping."""
    padded = np.pad(mask, 1, constant_values=False)
    interior = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1] &
                padded[1:-1, :-2] & padded[1:-1, 2:])
    return mask & ~interior


def composite_mask(pixels, mask, opacity=.5, color=(255, 82, 94), *, mode='fill', base=None):
    """Blend an aligned scalar mask into already-windowed pixels, never resize."""
    pixels, mask = np.asarray(pixels), np.asarray(mask)
    if mask.ndim != 2 or pixels.shape[:2] != mask.shape:
        raise ValueError('叠加掩膜必须与底图尺寸一致')
    if mode == 'difference' and (base is None or np.shape(base) != mask.shape):
        raise ValueError('差异模式需要两张尺寸一致的标量掩膜')
    rgb = (np.repeat(pixels[..., None], 3, axis=2) if pixels.ndim == 2 else pixels[..., :3]).copy()
    active = foreground(mask)
    if mode == 'outline':
        active = mask_boundary(active)
    if mode == 'difference':
        a = foreground(base)
        tint = np.zeros_like(rgb)
        tint[a & active] = (192, 203, 213)
        tint[a & ~active] = (255, 166, 69)
        tint[~a & active] = (193, 117, 255)
        # A missing auxiliary frame is unknown, never a negative mask vote.
        active = (a | active) & np.isfinite(mask) & np.isfinite(base)
    else:
        tint = np.asarray(color, dtype=np.uint8)
    alpha = np.clip(opacity, 0., 1.)
    if mode == 'difference':
        rgb[active] = np.rint(rgb[active]*(1-alpha) + tint[active]*alpha).astype(np.uint8)
    else:
        rgb[active] = np.rint(rgb[active]*(1-alpha) + tint*alpha).astype(np.uint8)
    return rgb


def aligned_sections(result, base_indices):
    """Align cross-section rows by real frame ID; absent/bad rows stay NaN."""
    rows, columns, indices, errors = result
    positions = {number: i for i, number in enumerate(indices)}
    out = [np.full((len(base_indices), plane.shape[1]), np.nan) for plane in (rows, columns)]
    for destination, number in enumerate(base_indices):
        source = positions.get(number)
        if source is not None:
            for target, plane in zip(out, (rows, columns)):
                target[destination] = plane[source]
    return (*out, tuple(base_indices), errors)
