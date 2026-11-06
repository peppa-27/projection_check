"""Editable label geometry, protected labels, real mouse gestures and exports."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import gc
from pathlib import Path
from threading import Event
import tempfile
import time
import unittest
from unittest.mock import patch
import sys

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from .app import ReviewWindow
from .data import Dataset, ImportOptions
from .geometry import ProjectionGeometry
from .labeling_core import Label, LabelSession, plane, brush_selection, polygon_selection
from .processing import ProcessingSource, ProcessingCancelled
from .test_geometry import write_case


class LabelCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.array = np.arange(120, dtype=np.int16).reshape(4, 5, 6)-40
        path = self.root/'raw.npy'
        np.save(path, self.array)
        self.dataset = Dataset([path], ImportOptions(domain='reconstruction'))
        self.session = LabelSession.load(ProcessingSource.from_dataset(self.dataset), Event())
        self.session.definitions.append(Label(2, str(2), (65, 220, 210), selected=False))

    def tearDown(self):
        self.session = None
        self.dataset.close()
        gc.collect()
        self.temp.cleanup()

    def test_plane_edits_share_one_volume_and_undo_redo(self):
        session = self.session
        for axis, index, target in [(0, 1, (2, 3)), (1, 2, (1, 3)), (2, 3, (1, 2))]:
            before = plane(session.labels, axis, index).copy()
            selection = np.zeros(before.shape, bool)
            selection[target] = True
            session.paint(axis, index, selection, 1)
            session.commit_plane(axis, index, before)
            self.assertEqual(session.labels[1, 2, 3], 1)
            session.undo()
            self.assertEqual(session.labels[1, 2, 3], 0)
            session.redo()
            self.assertEqual(session.labels[1, 2, 3], 1)
            session.undo()
        np.testing.assert_array_equal(self.dataset._array, self.array)

    def test_protection_erase_and_fast_brush_have_no_gaps(self):
        session = self.session
        session.labels[0, 1, 1] = 2
        selected = np.ones((5, 6), bool)
        session.paint(0, 0, selected, 1)
        self.assertEqual(session.labels[0, 1, 1], 2)
        session.definition(2).locked = True
        session.paint(0, 0, selected, 1, 'all')
        self.assertEqual(session.labels[0, 1, 1], 2)
        session.paint(0, 0, selected, 1, 'all', erase=True)
        self.assertEqual(session.labels[0, 1, 1], 2)
        self.assertEqual(np.count_nonzero(session.labels), 1)
        session.definition(1).locked = True
        session.paint(0, 1, selected, 1, 'all')
        self.assertFalse(session.labels[1].any())
        stroke = brush_selection((9, 20), (1.5, 4.5), (18.5, 4.5), .5)
        self.assertTrue(stroke[4, 1:19].all())
        self.assertEqual(stroke.sum(), 18)
        polygon = polygon_selection((9, 20), [(2, 2), (7, 2), (7, 7), (2, 7)])
        self.assertTrue(polygon[3, 3])
        self.assertFalse(polygon[0, 0])

    def test_threshold_is_raw_reversible_and_cancel_rolls_back_all_planes(self):
        session = self.session
        cancelled = Event()
        def progress(message):
            cancelled.set()
        with self.assertRaises(ProcessingCancelled):
            session.threshold(0, 'ge', 1, 'background', cancelled, progress)
        self.assertFalse(session.labels.any())
        self.assertFalse(session.undo_stack)
        session.threshold(0, 'ge', 1, 'background', Event())
        np.testing.assert_array_equal(session.labels, self.array >= 0)
        self.assertEqual(len(session.undo_stack), 1)
        session.undo()
        self.assertFalse(session.labels.any())
        session.redo()
        np.testing.assert_array_equal(session.labels, self.array >= 0)

    def test_selected_union_export_keeps_frame_ids_geometry_and_is_snapshot(self):
        session = self.session
        session.labels[1, 2, 3] = 1
        session.labels[2, 3, 4] = 2
        result = session.export_mask([1, 2], Event())
        try:
            self.assertEqual(result.indices, self.dataset.indices)
            self.assertEqual(result.geometry_paths, self.dataset.geometry_paths)
            self.assertTrue(result.mask and result.generated)
            expected = (session.labels != 0).astype(np.uint8)
            session.labels[:] = 0
            np.testing.assert_array_equal(result._array, expected)
        finally:
            result.close()
        with self.assertRaisesRegex(ValueError, '为空'):
            session.export_mask([1], Event())

    def test_save_restore_validates_origin_and_preserves_metadata(self):
        session = self.session
        session.threshold(0, 'ge', 2, 'background', Event())
        session.definition(2).name = '目标结构'
        session.definition(2).locked = True
        path = self.root/'labels.npz'
        session.save(path, Event())
        self.assertEqual(session.saved_revision, session.revision)
        expected = session.labels.copy()
        session.labels[:] = 0
        session.restore(path, Event())
        np.testing.assert_array_equal(session.labels, expected)
        self.assertEqual(session.definition(2).name, '目标结构')
        self.assertTrue(session.definition(2).locked)
        session.undo()
        self.assertFalse(session.labels.any())
        other_path = self.root/'other.npy'
        np.save(other_path, self.array)
        other = Dataset([other_path], ImportOptions(domain='reconstruction'))
        other_session = LabelSession.load(ProcessingSource.from_dataset(other), Event())
        try:
            with self.assertRaisesRegex(ValueError, '不匹配'):
                other_session.restore(path, Event())
        finally:
            del other_session
            other.close()


class LabelWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        _, self.slices = write_case(self.root)
        self.errors = []
        self.old_hook = sys.excepthook
        sys.excepthook = lambda *args: self.errors.append(args)
        self.window = ReviewWindow()
        self.window.show()
        self.window.import_paths([self.slices])
        self.wait()
        self.source = self.window.panes[3].dataset
        self.page = self.window.labeling_page
        self.window.select_page(2)
        self.page.source.setCurrentIndex(self.page.source.findData(self.source.id))
        self.page.open_button.click()
        self.wait()

    def wait(self):
        deadline = time.monotonic()+30
        while (self.window.jobs or any(p.orth_job or p._orth_requested for p in self.window.panes)) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertFalse(self.window.jobs)
        self.assertFalse(any(p.orth_job or p._orth_requested for p in self.window.panes))

    def tearDown(self):
        try:
            self.window.clear_all()
            self.wait()
            self.window.close()
            self.app.processEvents()
            gc.collect()
            self.assertFalse(self.errors, 'Qt callback failed')
        finally:
            sys.excepthook = self.old_hook
            self.temp.cleanup()

    def click_voxel(self, axis, x, y, mouse=Qt.MouseButton.LeftButton):
        view = self.page.views[axis]
        point = view.mapFromScene(QPointF(x+.5, y+.5))
        QTest.mouseClick(view.viewport(), mouse, Qt.KeyboardModifier.NoModifier, point)
        self.app.processEvents()

    def test_real_brush_erase_and_navigation_cross_planes(self):
        page = self.page
        self.assertEqual(self.window.pages.currentIndex(), 2)
        self.assertTrue(page.project_button.isEnabled())
        self.assertFalse(hasattr(self.window, 'processing_toggle'))
        original = self.source.frame(11).copy()
        page.tool_buttons['brush'].click()
        page.brush_size.setValue(1)
        self.click_voxel(0, 5, 3)
        self.assertEqual(page.session.labels[1, 3, 5], 1)
        self.assertTrue(page.undo_button.isEnabled())
        self.click_voxel(0, 5, 3, Qt.MouseButton.RightButton)
        self.assertEqual(page.session.labels[1, 3, 5], 0)
        page.undo_button.click()
        self.assertEqual(page.session.labels[1, 3, 5], 1)
        page.tool_buttons['navigate'].click()
        self.click_voxel(0, 5, 3)
        self.assertEqual(page.position, [1, 3, 5])
        self.assertEqual(page.spins[0].value(), 11)
        np.testing.assert_array_equal(self.source.frame(11), original)
        page.tool_buttons['brush'].click()
        self.click_voxel(1, 2, 0)
        self.assertEqual(page.session.labels[0, 3, 2], 1)
        page.history(False)
        self.assertEqual(page.session.labels[0, 3, 2], 0)

    def test_polygon_accept_paste_cancel_and_label_lock(self):
        page = self.page
        page.tool_buttons['polygon'].click()
        for x, y in [(1, 1), (5, 1), (5, 3), (1, 3)]:
            self.click_voxel(0, x, y)
        self.assertFalse(page.session.labels.any())
        page.accept_polygon()
        self.assertEqual(page.session.labels[1, 2, 2], 1)
        page.spins[0].setValue(12)
        page.paste_polygon()
        self.assertEqual(len(page.views[0].points), 4)
        page.accept_polygon()
        self.assertEqual(page.session.labels[2, 2, 2], 1)
        page.label_table.item(0, 3).setCheckState(Qt.CheckState.Checked)
        page.tool_buttons['erase'].click()
        page.brush_size.setValue(3)
        self.click_voxel(0, 2, 2)
        self.assertEqual(page.session.labels[2, 2, 2], 1)
        page.tool_buttons['polygon'].click()
        self.click_voxel(0, 0, 0)
        page.cancel_gestures()
        self.assertFalse(page.views[0].points)

    def test_polygon_closes_with_first_point_double_click_right_and_both_enter_keys(self):
        page = self.page
        self.window.activateWindow()
        page.tool_buttons['polygon'].click()
        for axis, points in enumerate(([(1, 1), (5, 1), (5, 3)],
                                        [(1, 0), (5, 0), (5, 2)],
                                        [(0, 0), (4, 0), (4, 2)])):
            view = page.views[axis]
            for closing in ('first', 'double', 'right', 'return', 'enter', 'button'):
                with self.subTest(axis=axis, closing=closing):
                    for x, y in points:
                        self.click_voxel(axis, x, y)
                    self.assertEqual(len(view.points), 3)
                    self.assertFalse(page.session.labels.any())
                    if closing == 'first':
                        self.click_voxel(axis, *points[0])
                    elif closing == 'double':
                        point = view.mapFromScene(QPointF(points[-1][0]+.5, points[-1][1]+.5))
                        QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton,
                                         Qt.KeyboardModifier.NoModifier, point)
                    elif closing == 'right':
                        self.click_voxel(axis, *points[-1], Qt.MouseButton.RightButton)
                    elif closing in ('return', 'enter'):
                        view.setFocus()
                        self.app.processEvents()
                        QTest.keyClick(view, Qt.Key.Key_Return if closing == 'return' else Qt.Key.Key_Enter)
                    else:
                        page.accept_polygon_button.click()
                    self.app.processEvents()
                    self.assertFalse(view.points)
                    self.assertTrue(page.session.labels.any())
                    page.undo_button.click()
                    self.assertFalse(page.session.labels.any())

    def test_first_polygon_vertex_can_still_be_dragged(self):
        page = self.page
        page.tool_buttons['polygon'].click()
        points = [(1, 1), (5, 1), (5, 3)]
        for x, y in points:
            self.click_voxel(0, x, y)
        view = page.views[0]
        start = view.mapFromScene(QPointF(1.5, 1.5))
        end = view.mapFromScene(QPointF(.5, 1.5))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        self.assertEqual(len(view.points), 3)
        self.assertFalse(page.session.labels.any())
        self.assertLess(view.points[0][0], 1)

    def test_slice_sliders_sparse_ids_zoom_and_hidden_pan_bars(self):
        page = self.page
        self.assertEqual(page.label_table.rowCount(), 1)
        self.assertLess(page.label_table.height(), 90)
        self.assertEqual(len(page.tool_buttons), 4)
        page.session.source = replace(page.session.source,
            sequence=replace(page.session.source.sequence, indices=(10, 13, 21)))
        page.sync_slice_controls()
        view = page.views[0]
        view._fit = False
        view.scale(4, 4)
        before = view.viewportTransform()
        page.slice_sliders[0].setValue(2)
        self.assertEqual(page.spins[0].value(), 21)
        self.assertEqual(page.position[0], 2)
        self.assertEqual(view.viewportTransform(), before)
        page.spins[0].setValue(13)
        self.assertEqual(page.slice_sliders[0].value(), 1)
        page.slice_sliders[1].setValue(4)
        page.slice_sliders[2].setValue(6)
        self.assertEqual(page.position, [1, 4, 6])
        page.navigate(0, (2.5, 3.5))
        self.assertEqual([s.value() for s in page.slice_sliders], [1, 3, 2])
        for v in page.views:
            self.assertFalse(v.horizontalScrollBar().isVisible())
            self.assertFalse(v.verticalScrollBar().isVisible())
        center = view.viewport().rect().center()
        QTest.mousePress(view.viewport(), Qt.MouseButton.MiddleButton, Qt.KeyboardModifier.NoModifier, center)
        QTest.mouseMove(view.viewport(), center+QPoint(30, 20))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.MiddleButton,
                           Qt.KeyboardModifier.NoModifier, center+QPoint(30, 20))
        self.assertNotEqual(view.viewportTransform(), before)
        self.assertFalse(page.session.labels.any())

    def test_threshold_preview_and_apply_keep_review_independent_and_project(self):
        page = self.page
        pane = self.window.panes[3]
        before = pane.view.pixmap_item.pixmap().toImage()
        page.threshold.setValue(30)
        page.threshold_preview.setChecked(True)
        self.assertFalse(page.session.labels.any())
        self.assertEqual(pane.view.pixmap_item.pixmap().toImage(), before)
        page.apply_threshold(True)
        self.wait()
        self.assertFalse(page.threshold_preview.isChecked())
        for pos, number in enumerate(self.source.indices):
            np.testing.assert_array_equal(page.session.labels[pos], self.source.frame(number) >= 30)
        page.export_button.click()
        self.wait()
        mask = self.window.selected_dataset()
        self.assertEqual((mask.domain, mask.mask, mask.indices), ('reconstruction', True, (10, 11, 12)))
        self.assertIs(pane.dataset, self.source)
        page.project_button.click()
        self.wait()
        result = self.window.selected_dataset()
        self.assertEqual((result.domain, result.mask, result.indices), ('projection', True, (20, 21)))
        self.assertEqual(result.shape, (100, 120))
        self.assertTrue(any(result.frame(n).any() for n in result.indices))
        geometry = ProjectionGeometry.from_dataset(mask)
        geometry.check_projection_dataset(result)
        self.assertGreater(self.window.panes[0].overlay_source.findData(result.id), 0)
        self.assertEqual(self.window.pages.currentIndex(), 2)
        self.assertIs(page.session, page.sessions[self.source.id])

    def test_selected_labels_export_and_switching_sources_preserve_edits(self):
        page = self.page
        page.session.labels[1, 2, 3] = 1
        page.add_label()
        page.session.labels[2, 3, 4] = 2
        page.label_table.item(1, 4).setCheckState(Qt.CheckState.Checked)
        page.output_scope.setCurrentIndex(1)
        page.export_labels()
        self.wait()
        result = self.window.selected_dataset()
        self.assertEqual(sum(int(result.frame(n).sum()) for n in result.indices), 2)
        old = page.session
        page.source.setCurrentIndex(page.source.findData(result.id))
        page.open_source()
        self.wait()
        self.assertEqual(np.count_nonzero(page.session.labels), 2)
        page.source.setCurrentIndex(page.source.findData(self.source.id))
        page.open_source()
        self.assertIs(page.session, old)
        self.assertEqual(np.count_nonzero(page.session.labels), 2)

    def test_clear_during_load_discards_late_session_and_close_protects_unsaved(self):
        page = self.page
        page.threshold.setValue(30)
        page.apply_threshold(True)
        self.wait()
        with patch('metal_seg_qt.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Cancel):
            self.window.close()
        self.assertTrue(self.window.isVisible())
        old_path = Path(page.session.storage.name)
        self.window.clear_all()
        self.wait()
        gc.collect()
        self.assertFalse(old_path.exists())
        self.assertIsNone(page.session)
        self.window.import_paths([self.slices])
        self.wait()
        key = next(iter(self.window.datasets))
        page.source.setCurrentIndex(page.source.findData(key))
        entered, release = Event(), Event()
        original_load = LabelSession.load
        def delayed(*args):
            entered.set()
            release.wait(3)
            return original_load(*args)
        with patch('metal_seg_qt.labeling_widget.LabelSession.load', side_effect=delayed):
            page.open_source()
            self.assertTrue(entered.wait(2))
            self.window.clear_all()
            release.set()
            self.wait()
        self.assertIsNone(page.session)
        self.assertFalse(page.sessions)

    def test_source_removal_protects_unsaved_labels_and_cleans_session(self):
        page = self.page
        page.threshold.setValue(30)
        page.apply_threshold(True)
        self.wait()
        with patch('metal_seg_qt.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Cancel):
            self.window.remove_selected()
        self.assertIn(self.source.id, self.window.datasets)
        with patch('metal_seg_qt.app.QMessageBox.question', return_value=QMessageBox.StandardButton.Discard):
            self.window.remove_selected()
        self.assertNotIn(self.source.id, self.window.datasets)
        self.assertIsNone(page.session)
        self.assertFalse(page.sessions)


if __name__ == '__main__':
    unittest.main()
