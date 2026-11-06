"""Percentile windows: independent NumPy reference and real selected-pane flow."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
from threading import Event
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtWidgets import QApplication

from .app import ReviewWindow
from .data import Dataset, ImportOptions
from .orthogonal import SequenceSpec
from .processing import ProcessingCancelled
from .windowing import IntensityStatistics, collect_statistics


class PercentileMathTests(unittest.TestCase):
    def test_percentiles_match_raw_numpy_and_constant_values_remain_centered(self):
        frame = np.array([[-32768, -15, -2, 0, 1, 9, 32767, np.nan, np.inf]])
        stats = IntensityStatistics.from_frame(frame)
        for lower, upper in ((0, 100), (1, 99), (12.345, 93.75), (40, 60)):
            expected = np.percentile(frame[np.isfinite(frame)], (lower, upper))
            low, high, width, level = stats.window(lower, upper)
            np.testing.assert_allclose((low, high), expected)
            self.assertAlmostEqual(width, expected[1]-expected[0])
            self.assertAlmostEqual(level, expected.mean())
        constant = IntensityStatistics.from_frame(np.full((3, 4), -100))
        self.assertEqual(constant.window(1, 99), (-100., -100., 1., -100.))
        for invalid in ((99, 1), (1, 1), (-1, 99), (1, 101)):
            with self.assertRaises(ValueError):
                stats.window(*invalid)
        with self.assertRaises(ValueError):
            IntensityStatistics.from_frame(np.full((2, 3), np.nan))

    def test_exact_global_histograms_for_uint16_signed_and_half_float(self):
        with tempfile.TemporaryDirectory() as folder:
            for dtype in ('uint8', 'uint16', 'int16', 'float16'):
                with self.subTest(dtype=dtype):
                    data = (np.arange(420).reshape(4, 7, 15)*7-100).astype(dtype)
                    if dtype == 'float16':
                        data[0, 0, :3] = [np.nan, np.inf, -np.inf]
                    path = Path(folder)/f'{dtype}.npy'
                    np.save(path, data)
                    dataset = Dataset([path], ImportOptions())
                    try:
                        stats = collect_statistics(SequenceSpec.from_dataset(dataset), Event())
                        self.assertFalse(stats.approximate)
                        self.assertEqual(stats.frames, 4)
                        self.assertEqual(stats.finite_count, int(np.isfinite(data).sum()))
                        for lower, upper in ((0, 100), (1, 99), (25.123, 80.001)):
                            expected = np.percentile(data[np.isfinite(data)].astype(np.float64), (lower, upper))
                            np.testing.assert_allclose(stats.window(lower, upper)[:2], expected)
                    finally:
                        dataset.close()

    def test_large_type_sampling_is_bounded_repeatable_and_cancellable(self):
        with tempfile.TemporaryDirectory() as folder:
            data = np.arange(3*17*19, dtype=np.float32).reshape(3, 17, 19)
            path = Path(folder)/'float.npy'
            np.save(path, data)
            dataset = Dataset([path], ImportOptions())
            try:
                spec = SequenceSpec.from_dataset(dataset)
                stats = collect_statistics(spec, Event(), sample_budget=120)
                repeat = collect_statistics(spec, Event(), sample_budget=120)
                self.assertTrue(stats.approximate)
                self.assertEqual(stats.finite_count, data.size)
                self.assertLessEqual(len(stats.values), 120)
                np.testing.assert_array_equal(stats.values, repeat.values)
                cancelled = Event()
                cancelled.set()
                with self.assertRaises(ProcessingCancelled):
                    collect_statistics(spec, cancelled)
            finally:
                dataset.close()


class PercentileGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.exceptions = []
        self.old_hook = sys.excepthook
        sys.excepthook = lambda *args: self.exceptions.append(repr(args[1]))
        self.window = ReviewWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        try:
            self.window.close()
            self.wait()
            self.assertFalse(self.exceptions, self.exceptions)
        finally:
            sys.excepthook = self.old_hook
            self.temp.cleanup()

    def wait(self):
        deadline = time.monotonic()+8
        while (self.window.jobs or self.window._percentile_timer.isActive()
               or any(p.orth_job or p._orth_requested for p in self.window.panes)):
            self.app.processEvents()
            time.sleep(.005)
            if time.monotonic() > deadline:
                self.fail(self.window.percentile_result.text())
        self.app.processEvents()

    def source(self, name, data, **options):
        path = self.root/f'{name}.npy'
        np.save(path, data)
        source = Dataset([path], ImportOptions(**options))
        self.window.add_generated_dataset(source)
        return source

    def test_live_current_view_uses_raw_pixels_and_only_changes_selected_bound_pane(self):
        data = np.arange(300, dtype=np.int16).reshape(3, 10, 10)
        source = self.source('projection', data)
        self.window.place(source, 0)
        self.window.place(source, 1)
        self.window.select_pane(0)
        pane, peer = self.window.panes[:2]
        peer.ww.setValue(12)
        peer.wl.setValue(3)
        pane.view.set_markers(selection=(0, 0, .1, .1))
        pane.view._fit = False
        pane.view.scale(2, 2)
        transform = pane.view.viewportTransform()
        self.window.percentile_lower.setValue(10)
        self.window.percentile_upper.setValue(90)
        self.wait()
        bounds = np.percentile(data[0], (10, 90))
        self.assertAlmostEqual(pane.ww.value(), bounds[1]-bounds[0])
        self.assertAlmostEqual(pane.wl.value(), bounds.mean())
        self.assertEqual((peer.ww.value(), peer.wl.value()), (12, 3))
        self.assertEqual(pane.view.viewportTransform(), transform)
        pane.set_layer(1)
        self.assertAlmostEqual(pane.wl.value(), bounds.mean())
        np.testing.assert_array_equal(source.frame(1), data[1])
        self.window.percentile_live.setChecked(False)
        pane.view_mode.setCurrentIndex(2)
        pane.row_spin.setValue(3)
        self.wait()
        self.window.percentile_lower.setValue(0)
        self.window.percentile_upper.setValue(100)
        self.window.apply_percentile_window()
        expected = data[:, 3, :]
        self.assertAlmostEqual(pane.ww.value(), float(expected.max()-expected.min()))
        self.assertAlmostEqual(pane.wl.value(), float((int(expected.max())+int(expected.min()))/2))

    def test_global_statistics_are_cached_and_threshold_preview_blocks_windowing(self):
        data = np.arange(300, dtype=np.int16).reshape(3, 10, 10)-150
        source = self.source('reconstruction', data, domain='reconstruction')
        self.window.place(source, 3)
        self.window.percentile_live.setChecked(False)
        self.window.percentile_scope.setCurrentIndex(1)
        self.window.percentile_lower.setValue(10)
        self.window.percentile_upper.setValue(90)
        self.window.apply_percentile_window()
        self.wait()
        pane = self.window.panes[3]
        expected = np.percentile(data, (10, 90))
        self.assertAlmostEqual(pane.ww.value(), expected[1]-expected[0])
        self.assertIn('精确统计', self.window.percentile_result.text())
        cached = self.window._percentile_statistics
        with patch('metal_seg_qt.app.collect_statistics', side_effect=AssertionError('must reuse cache')):
            self.window.percentile_lower.setValue(20)
            self.window.percentile_upper.setValue(80)
            self.window.apply_percentile_window()
            self.wait()
        expected = np.percentile(data, (20, 80))
        self.assertAlmostEqual(pane.wl.value(), expected.mean())
        self.assertAlmostEqual(pane.ww.value(), expected[1]-expected[0])
        self.assertIs(self.window._percentile_statistics, cached)
        pane.set_threshold_preview(0)
        self.window.update_percentile_actions()
        self.assertFalse(self.window.percentile_apply.isEnabled())
        pane.set_threshold_preview()
        self.window.update_percentile_actions()
        self.assertTrue(self.window.percentile_apply.isEnabled())
        mask = self.source('mask', (data > 0).astype(np.uint8), mask=True)
        self.window.place(mask, 0)
        self.assertFalse(self.window.percentile_apply.isEnabled())

    def test_clear_during_statistics_ignores_even_already_computed_result(self):
        source = self.source('projection', np.arange(100, dtype=np.int16).reshape(10, 10))
        self.window.place(source, 0)
        self.window.percentile_live.setChecked(False)
        self.window.percentile_scope.setCurrentIndex(1)
        entered, release = Event(), Event()
        stats = IntensityStatistics.from_frame(source.frame(0))
        def delayed(*args):
            entered.set()
            release.wait(3)
            return stats
        with patch('metal_seg_qt.app.collect_statistics', side_effect=delayed):
            try:
                self.window.apply_percentile_window()
                self.assertTrue(entered.wait(2))
                self.window.clear_all()
            finally:
                release.set()
            self.wait()
        self.assertIsNone(self.window._percentile_statistics)
        self.assertIsNone(self.window.percentile_job)
        self.assertFalse(self.window.percentile_apply.isEnabled())


if __name__ == '__main__':
    unittest.main()
