"""Whole-volume mask workflow, calibrated binary rays, and temporary lifetimes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
from pathlib import Path
from threading import Event
import gc
import tempfile
import unittest

import numpy as np
from .data import Dataset, ImportOptions
from .geometry import ProjectionGeometry, case_root
from .orthogonal import SequenceSpec, read_sections
from .processing import (ProcessingSource, ProcessingCancelled, threshold_mask, project_mask)
from .test_geometry import write_case


class MaskOperationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dataset(self, array, domain='reconstruction', mask=False):
        path = self.root / f'volume_{len(list(self.root.glob("*.npy")))}.npy'
        np.save(path, array)
        dataset = Dataset([path], ImportOptions(domain=domain, mask=mask))
        self.addCleanup(dataset.close)
        return dataset

    def test_threshold_uses_raw_signed_values_all_layers_and_excludes_nonfinite(self):
        array = np.array([[[-10, -1, 0], [1, 10, np.nan]],
                          [[-100, 7, np.inf], [3, -np.inf, 0]]], np.float32)
        original = self.dataset(array)
        source = ProcessingSource.from_dataset(original)
        for comparison, expected in [('ge', np.isfinite(array) & (array >= 0)),
                                     ('le', np.isfinite(array) & (array <= 0))]:
            result = threshold_mask(source, 0, comparison, Event())
            self.addCleanup(result.close)
            self.assertTrue(result.generated and result.mask)
            self.assertEqual(result.domain, 'reconstruction')
            self.assertEqual(result.dtype, 'uint8')
            self.assertEqual(result.indices, original.indices)
            for index in result.indices:
                np.testing.assert_array_equal(result.frame(index), expected[index])
        np.testing.assert_array_equal(original.frame(0), array[0])

    def native_mask(self, manufacturer=False):
        _, slices = write_case(self.root)
        if manufacturer:
            ini = slices / 'ImageParam.ini'
            ini.write_text(ini.read_text(encoding='utf-16') +
                           '\n[EquipmentModule]\nManufacturer=Fussen\n', encoding='utf-16')
        for index in (10, 11, 12):
            frame = np.zeros((5, 7), dtype='<i2')
            if index == 11:
                frame[3, 5] = 4000
            frame.tofile(slices / f'slice_{index}.slc')
        original = Dataset(list(slices.glob('*.slc')), ImportOptions())
        self.addCleanup(original.close)
        result = threshold_mask(ProcessingSource.from_dataset(original), 3000, 'ge', Event())
        self.addCleanup(result.close)
        return original, result, ProjectionGeometry.from_dataset(result)

    @staticmethod
    def analytic_single_voxel(geometry, matrix):
        # Independent per-ray slab test for the known occupied voxel, with
        # positive-length intersection (a corner-only contact is background).
        affine = np.eye(4)
        affine[:3, :3] = np.column_stack((geometry.column_step, geometry.row_step, geometry.slice_step))
        affine[:3, 3] = geometry.origin
        local = matrix @ affine
        inverse = np.linalg.inv(local[:, :3])
        source = -inverse @ local[:, 3]
        v, u = np.indices(geometry.detector_shape)
        directions = np.stack((u, v, np.ones_like(u)), axis=-1) @ inverse.T
        enter = np.zeros(geometry.detector_shape)
        leave = np.full(geometry.detector_shape, np.inf)
        valid = np.ones(geometry.detector_shape, bool)
        for axis, (low, high) in enumerate(((4.5, 5.5), (2.5, 3.5), (.5, 1.5))):
            direction = directions[..., axis]
            parallel = np.abs(direction) < 1e-14
            valid &= ~parallel | ((source[axis] >= low) & (source[axis] < high))
            safe = np.where(parallel, 1., direction)
            a, b = (low - source[axis]) / safe, (high - source[axis]) / safe
            enter = np.maximum(enter, np.where(parallel, -np.inf, np.minimum(a, b)))
            leave = np.minimum(leave, np.where(parallel, np.inf, np.maximum(a, b)))
        return valid & (leave > enter + 1e-12)

    def test_binary_projection_preserves_native_and_oblique_geometry_without_splat_holes(self):
        _, mask, geometry = self.native_mask(manufacturer=True)
        source = ProcessingSource.from_dataset(mask)
        angle = .3
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        oblique = replace(geometry, origin=rotation @ geometry.origin,
                          column_step=rotation @ geometry.column_step,
                          row_step=rotation @ geometry.row_step,
                          slice_step=rotation @ geometry.slice_step)
        for calibrated in (geometry, oblique):
            result = project_mask(source, calibrated, Event())
            self.addCleanup(result.close)
            self.assertEqual((result.domain, result.mask, result.indices), ('projection', True, (20, 21)))
            self.assertEqual(result.shape, (100, 120))
            self.assertEqual(case_root(result), self.root)
            for index, matrix in zip(result.indices, calibrated.matrices):
                expected = self.analytic_single_voxel(calibrated, matrix)
                self.assertGreater(expected.sum(), 10)  # A magnified voxel covers many detector pixels.
                np.testing.assert_array_equal(result.frame(index), expected)

    def test_grayscale_impostor_missing_layers_and_cancel_are_rejected(self):
        original, mask, geometry = self.native_mask()
        with self.assertRaisesRegex(ValueError, '不能使用普通'):
            project_mask(ProcessingSource.from_dataset(original), geometry, Event())
        original.mask = True  # Merely switching display mode does not make CT data binary.
        with self.assertRaisesRegex(ValueError, '非二值'):
            project_mask(ProcessingSource.from_dataset(original), geometry, Event())
        source = ProcessingSource.from_dataset(mask)
        incomplete = replace(source, sequence=replace(source.sequence, indices=(10, 12)))
        with self.assertRaisesRegex(ValueError, '缺层'):
            project_mask(incomplete, geometry, Event())
        cancelled = Event()
        cancelled.set()
        with self.assertRaises(ProcessingCancelled):
            project_mask(source, geometry, cancelled)

    def test_empty_masks_and_cancel_during_projection_return_no_partial_result(self):
        original, _, geometry = self.native_mask()
        empty = threshold_mask(ProcessingSource.from_dataset(original), 5000, 'ge', Event())
        self.addCleanup(empty.close)
        result = project_mask(ProcessingSource.from_dataset(empty), geometry, Event())
        self.addCleanup(result.close)
        self.assertFalse(any(result.frame(n).any() for n in result.indices))
        cancelled = Event()

        def progress(message):
            if message.startswith('生成投影掩膜'):
                cancelled.set()

        # Use a nonempty mask to reach a ray-tracing batch, then cancel it.
        original.mask = False
        mask = threshold_mask(ProcessingSource.from_dataset(original), 3000, 'ge', Event())
        self.addCleanup(mask.close)
        with self.assertRaises(ProcessingCancelled):
            project_mask(ProcessingSource.from_dataset(mask), geometry, cancelled, progress)

    def test_temporary_results_keep_real_layer_ids_and_live_background_readers(self):
        original, mask, _ = self.native_mask()
        path = mask.paths[0]
        spec = SequenceSpec.from_dataset(mask, cache_sections=True)
        original.close()
        mask.close()
        self.assertTrue(path.exists())
        rows, _, indices, errors = read_sections(spec, 3, 5, Event())
        self.assertFalse(errors)
        self.assertEqual(indices, (10, 11, 12))
        np.testing.assert_array_equal(rows[:, 5], [0, 1, 0])
        del spec
        gc.collect()
        self.assertFalse(path.exists())

    def test_unclaimed_result_closes_mapping_before_temporary_storage_cleanup(self):
        original = self.dataset(np.ones((2, 3, 4), dtype=np.int16))
        result = threshold_mask(ProcessingSource.from_dataset(original), 1, 'ge', Event())
        path = result.paths[0]
        self.assertTrue(path.exists())
        del result
        gc.collect()
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
