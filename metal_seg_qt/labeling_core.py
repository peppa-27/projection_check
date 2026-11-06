"""Disk-backed reconstruction labels, reversible plane edits and binary exports."""
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .processing import (TemporaryMaskStorage, check_cancelled, sequence_frames,
                         binary_threshold, finish_mask)


@dataclass
class Label:
    value: int
    name: str
    color: tuple
    visible: bool = True
    locked: bool = False
    selected: bool = True


def plane(array, axis, index):
    return (array[index, :, :] if axis == 0 else
            array[:, index, :] if axis == 1 else array[:, :, index])


class EditCommand:
    def __init__(self, root, name):
        self.storage = TemporaryDirectory(prefix='edit-', dir=root)
        self.name, self.entries = name, []

    def __del__(self):
        self.storage.cleanup()

    def record(self, axis, index, before, after):
        changed = np.flatnonzero(before.ravel() != after.ravel())
        if changed.size:
            path = Path(self.storage.name) / f'{len(self.entries)}.npz'
            np.savez(path, indices=changed, before=before.ravel()[changed],
                     after=after.ravel()[changed])
            self.entries.append((axis, index, path))

    def replay(self, labels, forward):
        entries = self.entries if forward else reversed(self.entries)
        for axis, index, path in entries:
            with np.load(path, allow_pickle=False) as data:
                plane(labels, axis, index).flat[data['indices']] = data['after' if forward else 'before']


class LabelSession:
    def __init__(self, source):
        if source.domain != 'reconstruction' or len(source.sequence.shape) != 2:
            raise ValueError('标签勾画需要重建域单通道数据')
        self.source = source
        self.storage = TemporaryDirectory(prefix='metal-review-labels-')
        self.volume = self.labels = None
        self.definitions = [Label(1, '金属', (255, 82, 94))]
        self.undo_stack, self.redo_stack = [], []
        self.revision = 0
        self.saved_revision = 0

    @classmethod
    def load(cls, source, cancelled, progress=None):
        session = cls(source)
        spec = source.sequence
        shape = (len(spec.indices), *spec.shape)
        session.labels = np.lib.format.open_memmap(Path(session.storage.name) / 'labels.npy',
            mode='w+', dtype=np.uint16, shape=shape)
        session.labels[:] = 0
        with sequence_frames(spec) as frames:
            for position, (number, frame) in enumerate(frames):
                check_cancelled(cancelled)
                if session.volume is None:
                    # SequenceSpec.dtype describes raw decoding. NPY / image
                    # readers retain their own signed / floating-point dtype.
                    session.volume = np.lib.format.open_memmap(Path(session.storage.name) / 'image.npy',
                        mode='w+', dtype=frame.dtype, shape=shape)
                if not np.can_cast(frame.dtype, session.volume.dtype, casting='safe'):
                    raise ValueError('切片的数据类型不一致，无法无损建立标注底图')
                session.volume[position] = frame
                if source.mask:
                    if not np.isin(frame, [0, 1, 255, 65535]).all():
                        raise ValueError(f'层 {number} 不是二值掩膜，请使用灰度原图或正确的二值掩膜')
                    session.labels[position] = frame != 0
                if progress:
                    progress(f'加载重建：{position+1}/{len(spec.indices)} 层')
        session.volume.flush()
        session.labels.flush()
        check_cancelled(cancelled)
        return session

    def __del__(self):
        for name in ('volume', 'labels'):
            mapped = getattr(self, name, None)
            if mapped is not None:
                mapped._mmap.close()
                setattr(self, name, None)
        if hasattr(self, 'storage'):
            self.storage.cleanup()

    def definition(self, value):
        return next(label for label in self.definitions if label.value == value)

    def editable(self, values, active, policy='background', erase=False):
        if self.definition(active).locked:
            return np.zeros(values.shape, bool)
        allowed = np.ones(values.shape, bool)
        for label in self.definitions:
            if label.locked:
                allowed &= values != label.value
        if erase:
            allowed &= values == active
        elif policy == 'background':
            allowed &= (values == 0) | (values == active)
        return allowed

    def paint(self, axis, index, selection, active, policy='background', erase=False):
        target = plane(self.labels, axis, index)
        target[selection & self.editable(target, active, policy, erase)] = 0 if erase else active

    @contextmanager
    def command(self, name):
        command = EditCommand(self.storage.name, name)
        try:
            yield command
        except BaseException:
            command.replay(self.labels, False)
            raise
        else:
            if command.entries:
                self.undo_stack.append(command)
                self.redo_stack.clear()
                # History is stored on disk; bound the number of gestures.
                del self.undo_stack[:-100]
                self.revision += 1

    def commit_plane(self, axis, index, before, name='笔画'):
        try:
            with self.command(name) as command:
                command.record(axis, index, before, plane(self.labels, axis, index))
        except BaseException:
            plane(self.labels, axis, index)[:] = before
            raise

    def undo(self):
        if self.undo_stack:
            command = self.undo_stack.pop()
            command.replay(self.labels, False)
            self.redo_stack.append(command)
            self.revision += 1

    def redo(self):
        if self.redo_stack:
            command = self.redo_stack.pop()
            command.replay(self.labels, True)
            self.undo_stack.append(command)
            self.revision += 1

    def threshold(self, threshold, comparison, active, policy, cancelled,
                  progress=None, axis=0, index=None):
        indices = range(self.labels.shape[axis]) if index is None else [index]
        with self.command('阈值辅助') as command:
            for offset, position in enumerate(indices):
                check_cancelled(cancelled)
                target = plane(self.labels, axis, position)
                before = target.copy()
                selection = binary_threshold(plane(self.volume, axis, position), threshold, comparison)
                self.paint(axis, position, selection, active, policy)
                try:
                    command.record(axis, position, before, target)
                except BaseException:
                    target[:] = before
                    raise
                if progress:
                    progress(f'应用阈值：{offset+1}/{len(indices)} 层')
            check_cancelled(cancelled)

    def export_mask(self, values, cancelled, progress=None):
        if not values or not set(values).issubset({label.value for label in self.definitions}):
            raise ValueError('请至少选择一个有效标签')
        storage = TemporaryMaskStorage(prefix='metal-review-label-mask-')
        path = Path(storage.name) / 'mask.npy'
        output = None
        try:
            output = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=self.labels.shape)
            nonempty = False
            for index in range(len(self.labels)):
                check_cancelled(cancelled)
                output[index] = np.isin(self.labels[index], values)
                nonempty |= bool(output[index].any())
                if progress:
                    progress(f'导出标签：{index+1}/{len(self.labels)} 层')
            if not nonempty:
                raise ValueError('所选标签为空，请先勾画或应用阈值')
            output.flush()
            output._mmap.close()
            output = None
            names = ' + '.join(self.definition(value).name for value in values)
            result = finish_mask(storage, path, self.source, self.source.sequence.indices,
                'reconstruction', f'{self.source.name} · {names} · v{self.revision}',
                f'标签勾画二值掩膜；标签 {list(values)} 合并；版本 {self.revision}。')
            storage = None
            return result
        finally:
            if output is not None:
                output._mmap.close()
            if storage is not None:
                storage.cleanup()

    def save(self, path, cancelled):
        check_cancelled(cancelled)
        # Write a sibling temporary file, then replace only after success.
        path = Path(path)
        with TemporaryDirectory(prefix='label-save-', dir=path.parent) as folder:
            temporary = Path(folder) / 'labels.npz'
            np.savez_compressed(temporary, labels=self.labels, indices=self.source.sequence.indices,
                metadata=json.dumps([asdict(label) for label in self.definitions], ensure_ascii=False),
                source_paths=json.dumps([str(p) for p in self.source.sequence.paths]))
            check_cancelled(cancelled)
            temporary.replace(path)
        self.saved_revision = self.revision

    def restore(self, path, cancelled):
        with np.load(path, allow_pickle=False) as data:
            values = data['labels']
            if (values.shape != self.labels.shape or values.dtype != np.uint16 or
                    tuple(data['indices']) != self.source.sequence.indices or
                    json.loads(str(data['source_paths'])) != [str(p) for p in self.source.sequence.paths]):
                raise ValueError('标签文件与当前重建数据的来源、尺寸或真实层号不匹配')
            definitions = [Label(**entry) for entry in json.loads(str(data['metadata']))]
            ids = [label.value for label in definitions]
            if (not ids or len(set(ids)) != len(ids) or any(not 1 <= value <= 65535 for value in ids)
                    or not np.isin(values, [0, *ids]).all()
                    or any(len(label.color) != 3 or any(not 0 <= c <= 255 for c in label.color)
                           for label in definitions)):
                raise ValueError('标签文件中的标签定义无效')
            with self.command('载入标签') as command:
                for index in range(len(values)):
                    check_cancelled(cancelled)
                    command.record(0, index, self.labels[index], values[index])
                    self.labels[index] = values[index]
                check_cancelled(cancelled)
            # Keep existing definitions for undo of older edits; imported IDs
            # replace matching definitions without orphaning existing history.
            imported = {label.value: label for label in definitions}
            self.definitions = [imported.pop(label.value, label) for label in self.definitions]
            self.definitions.extend(imported.values())


def brush_selection(shape, start, end, radius):
    """A capsule between sampled mouse positions leaves no gaps in fast strokes."""
    height, width = shape
    x0, y0 = start
    x1, y1 = end
    left, right = max(0, int(min(x0, x1)-radius)), min(width, int(max(x0, x1)+radius)+2)
    top, bottom = max(0, int(min(y0, y1)-radius)), min(height, int(max(y0, y1)+radius)+2)
    selection = np.zeros(shape, bool)
    y, x = np.ogrid[top:bottom, left:right]
    x, y = x+.5, y+.5
    length = (x1-x0)**2 + (y1-y0)**2
    t = np.clip(((x-x0)*(x1-x0)+(y-y0)*(y1-y0))/max(length, 1e-12), 0, 1)
    selection[top:bottom, left:right] = (x-x0-t*(x1-x0))**2 + (y-y0-t*(y1-y0))**2 <= radius**2
    return selection


def polygon_selection(shape, points):
    from PIL import Image, ImageDraw
    image = Image.new('1', (shape[1], shape[0]))
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=bool)
