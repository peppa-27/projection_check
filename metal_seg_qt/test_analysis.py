"""Analytical support counts, physical grids, residuals, and complete Qt workflow."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
import gc
import sys
import tempfile
import time
import unittest
import weakref

import numpy as np
from PIL import Image
from PySide6.QtCore import QPoint, Qt, QMimeData
from PySide6.QtGui import QDragEnterEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from .app import ReviewWindow, MIME_DATASET
from .data import Dataset, ImportOptions
from .geometry import ProjectionGeometry
from .processing import ProcessingSource, ProcessingCancelled, threshold_mask
from .analysis_core import (build_support, local_matrices, sampled_geometry, envelope,
    reproject, difference_counts, difference_pixels, ray_profile, roi_metrics,
    export_support, export_envelope, export_reprojection)
from .test_geometry import write_case


def mask_case(root):
    from consistency.consistency_filter import VolumeGrid, forward_project_envelope
    proj, slices = write_case(root)
    reconstruction = Dataset(list(slices.glob('*.slc')), ImportOptions())
    geometry = ProjectionGeometry.from_dataset(reconstruction)
    reconstruction.close()
    truth = np.zeros((3, 5, 7), bool)
    truth[1, 3, 5] = True
    masks = forward_project_envelope(truth, local_matrices(geometry, (20, 21), (100, 120)),
        VolumeGrid(truth.shape, (1., 1., 1.), (-.5, -.5, -.5)), (100, 120))
    for n, image in zip((20, 21), masks):
        Image.fromarray(image.astype(np.uint8)*255).save(proj/f'proj_{n}.png')
    return proj, slices, geometry


class AnalysisMathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proj, self.slices, self.geometry = mask_case(self.root)
        self.dataset = Dataset(list(self.proj.glob('*.png')), ImportOptions(mask=True))
        self.addCleanup(self.dataset.close)
        self.source = ProcessingSource.from_dataset(self.dataset)

    def support(self, geometry=None, factor=1, source=None):
        result = build_support(source or self.source, geometry or self.geometry, factor, Event())
        self.addCleanup(result.close)
        return result

    def projection(self, result, tau=.96, minimum=1, **kwargs):
        series = reproject(result, tau, minimum, Event(), **kwargs)
        self.addCleanup(series.array._mmap.close)
        return series

    def test_full_counts_equal_independent_per_voxel_projections_and_point_votes(self):
        result = self.support()
        for z, y, x in np.ndindex(result.shape):
            _, world = self.geometry.voxel(((x+.5)/7, (y+.5)/5), z+10)
            values = []
            for i, n in enumerate(result.indices):
                p = self.geometry.project(world, n, result.masks.shape[1:])
                values.append(int(result.masks[i, min(99, int(p[1]*100)), min(119, int(p[0]*120))]))
            self.assertEqual(result.hits[z, y, x], sum(values))
            self.assertEqual(result.valid[z, y, x], 2)
            self.assertAlmostEqual(result.score[z, y, x], sum(values)/2)
            states, _, _ = result.votes((x, y, z))
            self.assertEqual(states.tolist(), values)
        self.assertEqual(result.score[1, 3, 5], 1)

    def test_projection_ini_does_not_override_slice_geometry(self):
        (self.proj/'ImageParam.ini').write_text('[FileModule]\nBeginIndex=20\nEndIndex=21\n'
            '[RawImageInfo]\nWidth=120\nHeight=100\n', encoding='utf-8')
        geometry = ProjectionGeometry.from_projection_dataset(self.dataset)
        self.assertEqual(geometry.slice_shape, (5, 7))
        self.assertEqual((geometry.slice_begin, geometry.slice_end), (10, 12))
        np.testing.assert_array_equal(geometry.origin, self.geometry.origin)

    def test_invalid_views_are_excluded_and_no_visibility_is_unknown(self):
        matrices = self.geometry.matrices.copy()
        matrices[1, 0, 3] += 10000
        one = self.support(replace(self.geometry, matrices=matrices))
        self.assertTrue((one.valid == 1).all())
        matrices[0, 0, 3] += 10000
        none = self.support(replace(self.geometry, matrices=matrices))
        self.assertTrue((none.valid == 0).all())
        self.assertTrue(np.isnan(none.score).all())
        self.assertFalse(envelope(none, 0, 1).any())

    def test_missing_frames_resize_and_binary_validation(self):
        single = replace(self.source, sequence=replace(self.source.sequence,
            paths=self.source.sequence.paths[:1], indices=(20,)))
        result = self.support(source=single)
        self.assertTrue((result.valid == 1).all())
        small = self.root/'resized'
        small.mkdir()
        for n, path in zip((20, 21), self.source.sequence.paths):
            with Image.open(path) as image:
                image.resize((60, 50), Image.Resampling.NEAREST).save(small/f'frame_{n}.png')
        dataset = Dataset(list(small.iterdir()), ImportOptions(mask=True))
        self.addCleanup(dataset.close)
        resized = self.support(source=ProcessingSource.from_dataset(dataset))
        self.assertTrue((resized.valid == 2).all())
        with self.assertRaisesRegex(ValueError, '二值'):
            self.support(source=replace(self.source, mask=False))
        Image.fromarray(np.full((100, 120), 12, np.uint8)).save(self.proj/'proj_21.png')
        with self.assertRaisesRegex(ValueError, '非二值'):
            self.support()

    def test_sampled_grid_preserves_extent_signed_axes_and_exported_geometry(self):
        from itertools import product
        def corners(g):
            counts = (g.slice_shape[1], g.slice_shape[0], g.slice_end-g.slice_begin+1)
            return np.array([g.origin+x*g.column_step+y*g.row_step+z*g.slice_step
                for x, y, z in product(*[(-.5, n-.5) for n in counts])])
        for factor in (1, 2, 4):
            sampled = sampled_geometry(self.geometry, factor)
            np.testing.assert_allclose(corners(sampled), corners(self.geometry))
        result = self.support(factor=2)
        dataset = export_support(result)
        self.addCleanup(dataset.close)
        self.assertFalse(dataset.mask)
        self.assertEqual(dataset.domain, 'reconstruction')
        self.assertIs(ProjectionGeometry.from_dataset(dataset), result.geometry)
        # The ordinary threshold workflow must retain this sampled physical grid.
        mask = threshold_mask(ProcessingSource.from_dataset(dataset), .5, 'ge', Event())
        self.addCleanup(mask.close)
        self.assertIs(ProjectionGeometry.from_dataset(mask), result.geometry)

    def test_reprojection_priority_residuals_roi_and_export(self):
        from consistency.consistency_filter import VolumeGrid, forward_project_envelope
        result = self.support()
        updates = []
        series = self.projection(result, partial=lambda data: updates.append(data[1]) if data[0] == 'view' else None,
                                 priority=lambda: 1)
        self.assertEqual(updates, [1, 0])
        self.assertTrue(series.ready.all())
        expected = forward_project_envelope(envelope(result, .96, 1), result.matrices,
            VolumeGrid(result.shape, (1., 1., 1.), (-.5, -.5, -.5)), result.masks.shape[1:])
        np.testing.assert_array_equal(series.array, expected)
        for i in range(2):
            np.testing.assert_array_equal(series.counts[i], difference_counts(result.masks[i], series.array[i]))
        a = np.array([[1, 1], [0, 0]], np.uint8)
        b = np.array([[1, 0], [1, 0]], np.uint8)
        self.assertEqual(difference_counts(a, b).tolist(), [1, 1, 1])
        np.testing.assert_array_equal(difference_pixels(a, b)[0, 1], [255, 166, 69])
        np.testing.assert_array_equal(difference_pixels(a, b)[1, 0], [193, 117, 255])
        self.assertEqual(difference_counts(a, b, (0, 0, .5, .5)).tolist(), [1, 0, 0])
        self.assertEqual(difference_counts(np.ones((2, 2)), np.ones((2, 2)), (1, 1, 1, 1)).tolist(), [1, 0, 0])
        counts = roi_metrics(series, (0, 0, .5, .5), Event())
        for i in range(2):
            np.testing.assert_array_equal(counts[i], difference_counts(result.masks[i], series.array[i], (0, 0, .5, .5)))
        volume = export_envelope(result, .96, 1, Event())
        projection = export_reprojection(series)
        self.addCleanup(volume.close)
        self.addCleanup(projection.close)
        self.assertEqual(volume.indices, (10, 11, 12))
        self.assertEqual(projection.indices, (20, 21))
        np.testing.assert_array_equal(volume.frame(11), envelope(result, .96, 1)[1])

    def test_ray_curve_traverses_known_voxel_and_cancellation(self):
        result = self.support()
        point = self.geometry.project(self.geometry.voxel((5.5/7, 3.5/5), 11)[1], 20, (100, 120))
        ray = result.geometry.backproject(point, 20, result.masks.shape[1:])
        distance, score, cells = ray_profile(result, ray)
        self.assertTrue((np.diff(distance) > 0).all())
        self.assertIn([5, 3, 1], cells.tolist())
        for value, (x, y, z) in zip(score, cells):
            self.assertEqual(value, result.score[z, y, x])
        cancelled = Event()
        cancelled.set()
        with self.assertRaises(ProcessingCancelled):
            build_support(self.source, self.geometry, 1, cancelled)
        with self.assertRaises(ProcessingCancelled):
            reproject(result, .96, 1, cancelled)

    def test_native_fussen_frame_is_used_for_support_and_export(self):
        ini = self.slices/'ImageParam.ini'
        ini.write_text(ini.read_text(encoding='utf-16')+'\n[EquipmentModule]\nManufacturer=Fussen\n', encoding='utf-16')
        geometry = ProjectionGeometry.from_projection_dataset(self.dataset)
        np.testing.assert_allclose(geometry.origin, self.geometry.origin*[-1, 1, -1])
        result = self.support(geometry)
        for z, y, x in np.ndindex(result.shape):
            votes, _, good = result.votes((x, y, z))
            self.assertEqual(result.hits[z, y, x], (votes == 1).sum())
            self.assertEqual(result.valid[z, y, x], good.sum())

    def test_input_frame_order_is_sorted_by_real_ids_without_changing_results(self):
        expected = self.support()
        reversed_source = replace(self.source, sequence=replace(self.source.sequence,
            paths=self.source.sequence.paths[::-1], indices=self.source.sequence.indices[::-1]))
        result = self.support(source=reversed_source)
        self.assertEqual(result.indices, (20, 21))
        np.testing.assert_array_equal(result.masks, expected.masks)
        np.testing.assert_array_equal(result.hits, expected.hits)
        np.testing.assert_array_equal(result.valid, expected.valid)

    def test_cancel_after_partial_projection_releases_its_temporary_mapping(self):
        result = self.support()
        cancelled = Event()
        published = []
        def partial(data):
            if data[0] == 'series':
                published.extend([weakref.ref(data[1]), data[1].path])
            elif data[0] == 'view':
                cancelled.set()
        with self.assertRaises(ProcessingCancelled):
            reproject(result, .96, 1, cancelled, partial=partial)
        gc.collect()
        self.assertIsNone(published[0]())
        self.assertFalse(published[1].exists())


class AnalysisWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.proj, self.slices, _ = mask_case(self.root)
        self.exceptions = []
        self.old_hook = sys.excepthook
        sys.excepthook = lambda cls, value, tb: self.exceptions.append(repr(value))
        self.window = ReviewWindow()
        self.window.show()
        self.page = self.window.reconstruction_page
        self.window.select_page(1)
        self.window.import_paths([self.proj], target=-2)
        self.wait()

    def wait(self):
        deadline = time.monotonic()+30
        while (self.window.jobs or self.page.jobs or self.page.pending_projection
                or self.page._projection_timer.isActive() or self.page._metrics_timer.isActive()):
            self.app.processEvents()
            time.sleep(.003)
            if time.monotonic() > deadline:
                self.fail('Timed out: '+self.page.status.text())
        for _ in range(2):
            self.app.processEvents()
        self.assertFalse(self.exceptions, self.exceptions)

    def tearDown(self):
        try:
            self.window.close()
            self.wait()
            gc.collect()
            self.temp.cleanup()
        finally:
            sys.excepthook = self.old_hook

    def build(self):
        self.page.resolution.setCurrentIndex(2)
        self.page.start_support()
        self.wait()
        self.assertIsNotNone(self.page.result, self.page.status.text())
        self.assertIsNotNone(self.page.series, self.page.status.text())

    def test_independent_page_point_votes_ray_roi_and_export_full_workflow(self):
        self.assertTrue(all(p.dataset is None for p in self.window.panes))
        self.build()
        self.assertTrue(all(size > 100 for size in self.page.charts_splitter.sizes()))
        self.page.pick_volume((5.5/7, 3.5/5))
        self.assertEqual(self.page.selected_votes.tolist(), [1, 1])
        self.assertIn('100.00%', self.page.voxel_info.text())
        rect = self.page.error_plot.plot_rect()
        QTest.mouseClick(self.page.error_plot, Qt.MouseButton.LeftButton,
                        pos=QPoint(int(self.page.error_plot.px(1)), int(rect.center().y())))
        self.assertEqual(self.page.view_spin.value(), 21)
        self.page.view_spin.setValue(20)
        self.page.pick_projection((70.5/120, 58/100))
        self.assertIsNotNone(self.page.ray)
        self.assertGreater(len(self.page.profile[2]), 0)
        self.page.select_ray_cell(0)
        self.assertIsNotNone(self.page.selected_xyz)
        self.page.sync_markers(0, None, (0, 0, .5, .5))
        self.wait()
        self.assertIn('框内', self.page.summary.text())
        self.page.export('score')
        self.page.export('volume')
        self.wait()
        self.page.export('projection')
        self.assertEqual(len(self.window.datasets), 4)
        self.assertTrue(all(p.dataset is None for p in self.window.panes))
        result = self.page.result
        self.window.select_page(0)
        self.window.select_page(1)
        self.assertIs(self.page.result, result)
        paths = list(result.paths.values())
        del result
        self.window.clear_all()
        self.wait()
        gc.collect()
        self.assertTrue(all(not path.exists() for path in paths))

    def test_parameter_changes_replace_reprojection_and_keep_support_and_native_ids(self):
        self.build()
        source = self.page.result
        self.page.tau.setValue(.5)
        self.page.tau.setValue(1)
        self.wait()
        self.assertIs(self.page.result, source)
        self.assertEqual(self.page.series.threshold, 1)
        self.assertEqual(self.page.result.indices, (20, 21))
        self.page.orientation.setCurrentIndex(1)
        self.assertEqual(self.page.volume_view.pixmap_item.pixmap().height(), 3)
        self.page.orientation.setCurrentIndex(2)
        self.assertEqual(self.page.volume_view.pixmap_item.pixmap().width(), 5)

    def test_clear_during_build_cancels_and_ignores_late_results(self):
        self.page.start_support()
        self.window.clear_all()
        self.wait()
        self.assertIsNone(self.page.result)
        self.assertIsNone(self.page.series)
        self.assertEqual(self.page.source_combo.count(), 1)

    def test_volume_maximize_restores_layout_and_keeps_analysis_and_marks(self):
        self.build()
        page = self.page
        page.charts_splitter.setSizes([400, 700])
        self.app.processEvents()
        sizes = page.charts_splitter.sizes()
        view_size = page.volume_view.viewport().size()
        page.volume_view._fit = False
        page.volume_view.scale(2, 2)
        transform = page.volume_view.transform()
        page.volume_view.set_markers(selection=(.2, .2, .7, .8))
        result, series, plane = page.result, page.series, page.plane_spin.value()
        page.volume_maximize_button.click()
        self.app.processEvents()
        self.assertTrue(page.volume_maximized)
        self.assertFalse(page.projection_panel.isVisible())
        self.assertFalse(page.charts_panel.isVisible())
        self.assertTrue(page.source_combo.isVisible())
        self.assertGreater(page.volume_view.viewport().width(), view_size.width())
        self.assertGreater(page.volume_view.viewport().height(), view_size.height())
        self.window.select_page(0)
        self.window.select_page(1)
        self.assertTrue(page.volume_maximized)
        self.window.activateWindow()
        page.volume_view.setFocus()
        self.app.processEvents()
        QTest.keyClick(page.volume_view, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertFalse(page.volume_maximized)
        self.assertTrue(page.projection_panel.isVisible())
        self.assertTrue(page.charts_panel.isVisible())
        self.assertEqual(page.charts_splitter.sizes(), sizes)
        self.assertEqual(page.volume_view.transform(), transform)
        self.assertEqual(page.volume_view.selection, (.2, .2, .7, .8))
        self.assertEqual(page.plane_spin.value(), plane)
        self.assertIs(page.result, result)
        self.assertIs(page.series, series)
        page.volume_maximize_button.click()
        page.reset_analysis()
        self.assertFalse(page.volume_maximized)
        self.assertTrue(page.projection_panel.isVisible())

    def test_source_dropdown_selects_masks_and_drag_entry_is_disabled(self):
        source = self.page.current_source()
        self.page.source_combo.setCurrentIndex(0)
        self.assertIsNone(self.page.current_source())
        self.assertFalse(self.page.build_button.isEnabled())
        self.page.source_combo.setCurrentIndex(self.page.source_combo.findData(source.id))
        self.assertIs(self.page.current_source(), source)
        self.assertTrue(self.page.build_button.isEnabled())
        self.assertFalse(self.window.reconstruction_tab.acceptDrops())
        self.assertTrue(all(not view.acceptDrops() for view in self.page.projection_views))
        mime = QMimeData()
        mime.setData(MIME_DATASET, source.id.encode())
        enter = QDragEnterEvent(QPoint(10, 10), Qt.DropAction.CopyAction, mime,
                               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(self.window.reconstruction_tab, enter)
        self.assertFalse(enter.isAccepted())
        self.assertEqual(self.page.current_source().id, source.id)

    def test_error_curves_do_not_fill_over_and_hide_each_other(self):
        plot = self.page.error_plot
        plot.set_data((0, 1, 2), [(np.array([.75, .75, .75]), '#ffa645'),
                                (np.array([.2, .95, .2]), '#c175ff')])
        self.app.processEvents()
        image = plot.grab().toImage()
        x = int(plot.px(1))
        y = int(plot.plot_rect().bottom()-.75*plot.plot_rect().height())
        colors = [image.pixelColor(x, yy) for yy in range(max(0, y-2), min(image.height(), y+3))]
        self.assertTrue(any(c.red() > 200 and 80 < c.green() < 210 and c.blue() < 140 for c in colors))


if __name__ == '__main__':
    unittest.main()
