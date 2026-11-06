"""Run with: python -m unittest metal_seg_qt.test_review -v"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QWheelEvent
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest

from .app import DISPLAY_PANES_ROLE, MIME_DATASET, ImageView, ReviewWindow
from .data import (Dataset, ImportOptions, auto_window, discover_groups,
                   display_pixels, load_datasets)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        # Metadata lookup includes a dataset's parent. Keep that parent inside
        # the fixture, away from any unrelated INI/XML in the system temp dir.
        self.root = Path(self.temp.name) / 'case'
        self.root.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def images(self, folder='png', numbers=(1, 2, 10)):
        directory = self.root / folder
        directory.mkdir()
        for number in numbers:
            Image.fromarray(np.full((5, 7), number, dtype=np.uint8)).save(directory / f'proj_{number}.png')
        return directory

    def test_real_layer_ids_natural_sort_and_missing_layer(self):
        source = Dataset(list(self.images().iterdir()), ImportOptions())
        self.assertEqual(source.indices, (1, 2, 10))
        self.assertEqual(source.frame(10)[0, 0], 10)
        with self.assertRaises(KeyError):
            source.frame(3)
        source.close()

    def test_raw_xml_shape_manual_dtype_and_bad_size(self):
        folder = self.root / 'proj'
        folder.mkdir()
        (folder / 'cbct_task.xml').write_text(
            '<?xml version="1.0" encoding="utf-16"?><root>'
            '<Sensor_Vertical_Size v="3"/><Sensor_Horizontal_Size v="4"/></root>', encoding='utf-8')
        expected = np.arange(12, dtype='<u2').reshape(3, 4)
        expected.tofile(folder / 'proj_5.prj')
        source = Dataset([folder / 'proj_5.prj'], ImportOptions())
        np.testing.assert_array_equal(source.frame(5), expected)
        source.close()
        with self.assertRaisesRegex(ValueError, '不匹配'):
            Dataset([folder / 'proj_5.prj'], ImportOptions(dtype='<f4'))
        source = Dataset([folder / 'proj_5.prj'], ImportOptions(use_xml=False, height=3, width=4, dtype='<f2'))
        self.assertEqual(source.frame(5).dtype, np.dtype('<f2'))
        source.close()

    def test_memory_mapped_volumes_release_file(self):
        expected = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
        for extension in ('.npy', '.bin'):
            path = self.root / ('volume' + extension)
            if extension == '.npy':
                np.save(path, expected)
            else:
                expected.tofile(path)
            source = Dataset([path], ImportOptions(height=4, width=5, dtype='<f4', use_xml=False))
            np.testing.assert_array_equal(source.frame(2), expected[2])
            self.assertEqual(source.indices, (0, 1, 2))
            source.close()
            path.unlink()  # Windows would reject this if a mapped file was still open.

    def test_window_nan_infinity_rgb_and_binary_mask(self):
        frame = np.array([[0, 50, 100, np.nan, np.inf, -np.inf]])
        np.testing.assert_array_equal(display_pixels(frame, 100, 50), [[0, 128, 255, 0, 255, 0]])
        np.testing.assert_array_equal(display_pixels(frame, 100, 50, True), [[0, 255, 255, 0, 0, 0]])
        for frame in (np.zeros((2, 2)), np.ones((2, 2)), np.full((2, 2), np.nan)):
            width, level = auto_window(frame)
            self.assertGreater(width, 0)
            self.assertTrue(np.isfinite(level))
        rgb = np.array([[[255, 0, 128]]], dtype=np.uint8)
        np.testing.assert_array_equal(display_pixels(rgb, 1, 0), rgb)

    def test_auto_window_uses_only_rectangle_pixels(self):
        frame = np.full((100, 100), 60000., dtype=np.float32)
        frame[20:40, 30:50] = np.arange(400).reshape(20, 20)
        self.assertEqual(auto_window(frame, (.3, .2, .5, .4)), auto_window(frame[20:40, 30:50]))
        self.assertNotEqual(auto_window(frame), auto_window(frame, (.3, .2, .5, .4)))
        # A zero-width rectangle still covers one pixel column.
        self.assertEqual(auto_window(frame, (.3, .2, .3, .4)), auto_window(frame[20:40, 30:31]))

    def test_slc_is_reconstruction_and_infers_variable_square_sizes(self):
        for side in (800, 600):
            folder = self.root / f'volume_{side}'
            folder.mkdir()
            path = folder / 'slice_0.slc'
            np.zeros((side, side), dtype='<u2').tofile(path)
            source = Dataset([path], ImportOptions())
            self.assertEqual(source.domain, 'reconstruction')
            self.assertEqual(source.shape, (side, side))
            source.close()

    def test_slice_raw_preserves_signed_values_independent_of_other_raw_dtype(self):
        expected = np.array([-32768, -2048, -1000, -1, 0, 1, 1024, 4095, 32767],
                            dtype='<i2').reshape(3, 3)
        slc = self.root / 'slice_0.slc'
        expected.tofile(slc)
        # A float32 setting for other Raw files must not affect SLC decoding
        # or the number of bytes used to infer the square dimensions.
        source = Dataset([slc], ImportOptions(dtype='<f4'))
        self.assertEqual(source.dtype, 'int16')
        np.testing.assert_array_equal(source.frame(0), expected)
        source.close()
        folder = self.root / 'slice'
        folder.mkdir()
        raw = folder / 'volume.raw'
        expected.tofile(raw)
        source = Dataset([raw], ImportOptions(dtype='<f4', auto_reconstruction_shape=False,
            reconstruction_height=3, reconstruction_width=3))
        self.assertEqual(source.dtype, 'int16')
        np.testing.assert_array_equal(source.frame(0), expected)
        source.close()

    def test_mixed_sample_uses_unsigned_projection_and_signed_slice(self):
        proj = self.root / 'proj'
        proj.mkdir()
        slices = self.root / 'slice'
        slices.mkdir()
        projection = np.full((3, 3), 65535, dtype='<u2')
        reconstruction = np.full((3, 3), -1000, dtype='<i2')
        projection.tofile(proj / 'proj_0.prj')
        reconstruction.tofile(slices / 'slice_0.slc')
        sources, errors = load_datasets([self.root], ImportOptions(height=3, width=3, use_xml=False))
        self.assertFalse(errors)
        try:
            by_domain = {s.domain: s for s in sources}
            self.assertEqual(by_domain['projection'].dtype, 'uint16')
            self.assertEqual(by_domain['reconstruction'].dtype, 'int16')
            np.testing.assert_array_equal(by_domain['projection'].frame(0), projection)
            np.testing.assert_array_equal(by_domain['reconstruction'].frame(0), reconstruction)
        finally:
            for source in sources:
                source.close()

    def test_reconstruction_ini_dimensions_override_projection_size_and_manual_rectangles(self):
        folder = self.root / 'slice'
        folder.mkdir()
        path = folder / 'slice_0.slc'
        np.arange(21, dtype='<u2').reshape(3, 7).tofile(path)
        ini = folder / 'ImageParam.ini'
        ini.write_text('[ImageModule]\nHeight=3\nWidth=7\n', encoding='utf-16')
        source = Dataset([path], ImportOptions(height=768, width=768))
        self.assertEqual(source.shape, (3, 7))
        self.assertEqual(source.shape_source, 'ImageParam.ini')
        source.close()
        ini.unlink()
        source = Dataset([path], ImportOptions(auto_reconstruction_shape=False,
            reconstruction_height=3, reconstruction_width=7))
        self.assertEqual(source.shape, (3, 7))
        source.close()
        with self.assertRaisesRegex(ValueError, '无法推导 SLC'):
            Dataset([path], ImportOptions())

    def test_slice_image_folder_is_reconstruction_with_its_own_dimensions(self):
        folder = self.root / 'slice'
        folder.mkdir()
        Image.fromarray(np.zeros((80, 120), np.uint8)).save(folder / 'image_0.png')
        sources, errors = load_datasets([folder], ImportOptions())
        self.assertFalse(errors)
        self.assertEqual(sources[0].domain, 'reconstruction')
        self.assertEqual(sources[0].shape, (80, 120))
        sources[0].close()

    def test_sample_folder_and_partial_failure(self):
        self.images('original')
        self.images('mask')
        self.assertEqual(len(discover_groups([self.root])), 2)
        bad = self.root / 'bad.npy'
        np.save(bad, np.empty((1, 2, 3, 4)))
        sources, errors = load_datasets([self.root / 'original', bad], ImportOptions())
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(errors), 1)
        for source in sources:
            source.close()

    def test_folder_import_keeps_root_images_and_direct_subfolders_separate(self):
        Image.fromarray(np.zeros((5, 7), np.uint8)).save(self.root / 'overview.png')
        original = self.images('original')
        mask = self.images('mask')
        deeper = original / 'not_scanned'
        deeper.mkdir()
        Image.fromarray(np.zeros((5, 7), np.uint8)).save(deeper / 'proj_0.png')
        groups = discover_groups([self.root, self.root])
        self.assertEqual(len(groups), 3)  # Repeated folder drop is deduplicated.
        self.assertEqual({group[0].parent for group in groups}, {self.root, original, mask})
        self.assertFalse(any(path.parent == deeper for group in groups for path in group))

    def test_duplicate_ids_and_mismatched_slice_shape(self):
        folder = self.images(numbers=(1, 2))
        Image.fromarray(np.zeros((3, 3), np.uint8)).save(folder / 'proj_2.png')
        source = Dataset(list(folder.iterdir()), ImportOptions())
        with self.assertRaisesRegex(ValueError, '不一致'):
            source.frame(2)
        source.close()
        Image.fromarray(np.zeros((5, 7), np.uint8)).save(folder / 'mask_1.png')
        with self.assertRaisesRegex(ValueError, '重复层号'):
            Dataset(list(folder.iterdir()), ImportOptions())

    def test_multipage_tiff_and_safe_npy(self):
        pages = [Image.fromarray(np.full((3, 4), i, np.uint16)) for i in range(3)]
        path = self.root / 'stack.tiff'
        pages[0].save(path, save_all=True, append_images=pages[1:])
        source = Dataset([path], ImportOptions())
        self.assertEqual(source.indices, (0, 1, 2))
        self.assertEqual(source.frame(2)[0, 0], 2)
        source.close()
        path = self.root / 'object.npy'
        np.save(path, np.array([{'unsafe': 'object'}], dtype=object))
        with self.assertRaises(ValueError):
            Dataset([path], ImportOptions())


class ViewStabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def gesture_view(self):
        view = ImageView()
        self.addCleanup(view.close)
        view.resize(400, 320)
        view.show()
        self.app.processEvents()
        view.show_pixels(np.zeros((128, 128), np.uint8), reset=True)
        return view

    def test_press_defers_click_and_drag_draws_rectangle_without_crosshair(self):
        view = self.gesture_view()
        points, marks = [], []
        view.pointPicked.connect(points.append)
        view.markersChanged.connect(lambda cross, rect: marks.append((cross, rect)))
        start, end = view.mapFromScene(QPointF(20, 20)), view.mapFromScene(QPointF(90, 100))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        self.assertIsNone(view.crosshair)
        self.assertEqual(points, [])
        self.assertEqual(marks, [])
        QTest.mouseMove(view.viewport(), end)
        self.assertIsNotNone(view.selection)
        self.assertIsNone(view.crosshair)
        self.assertTrue(all(cross is None for cross, _ in marks))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertEqual(points, [])
        self.assertIsNotNone(view.selection)
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        self.assertEqual(points, [])
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        self.assertEqual(len(points), 1)
        self.assertIsNotNone(view.crosshair)
        self.assertIsNone(view.selection)

    def test_long_hold_is_not_a_click_and_small_drag_after_hold_selects(self):
        view = self.gesture_view()
        points = []
        view.pointPicked.connect(points.append)
        start = view.mapFromScene(QPointF(30, 30))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.qWait(QApplication.startDragTime() + 30)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        self.assertEqual(points, [])
        self.assertIsNone(view.crosshair)
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.qWait(QApplication.startDragTime() + 30)
        end = start + QPoint(2, 2)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertEqual(points, [])
        self.assertIsNotNone(view.selection)
        self.assertIsNone(view.crosshair)

    def test_repeated_redraws_preserve_exact_zoom_and_pixel_position(self):
        view = ImageView()
        self.addCleanup(view.close)
        view.show()
        pixels = np.zeros((768, 768), dtype=np.uint8)
        for size, zoom in [((556, 302), 1.44), ((556, 302), 4.299817),
                           ((557, 303), 2.0736), ((558, 304), 1.2)]:
            with self.subTest(size=size, zoom=zoom):
                view.resize(*size)
                self.app.processEvents()
                view.show_pixels(pixels, reset=True)
                view._fit = False
                view.scale(zoom, zoom)
                view.centerOn(340.25, 390.75)
                view.notify_view_changed()
                before = view.viewportTransform()
                scrolls = (view.horizontalScrollBar().value(), view.verticalScrollBar().value())
                for _ in range(100):
                    view.show_pixels(pixels)
                self.assertEqual(view.viewportTransform(), before)
                self.assertEqual((view.horizontalScrollBar().value(),
                                  view.verticalScrollBar().value()), scrolls)

    def test_missing_frames_preserve_geometry_and_pending_linked_zoom(self):
        view = ImageView()
        self.addCleanup(view.close)
        view.resize(556, 302)
        view.show()
        self.app.processEvents()
        pixels = np.zeros((768, 768), dtype=np.uint8)
        view.show_pixels(pixels, reset=True)
        view._fit = False
        view.scale(2.0736, 2.0736)
        view.centerOn(340.25, 390.75)
        view.notify_view_changed()
        before = view.viewportTransform()
        for _ in range(100):
            view.show_pixels(None)
            self.assertFalse(view.has_image())
            self.assertFalse(view.pixmap_item.isVisible())
            self.assertIsNone(view.image_position(QPointF(100, 100)))
            view.show_pixels(pixels)
        self.assertEqual(view.viewportTransform(), before)
        view.show_pixels(None)
        state = (3.2, .6, .55, False)
        view.apply_view_state(state)
        self.assertEqual(view.view_state(), state)
        view.show_pixels(pixels)
        self.assertAlmostEqual(view.view_state()[0], 3.2, places=6)
        restored = view.viewportTransform()
        for _ in range(100):
            view.show_pixels(None)
            view.show_pixels(pixels)
        self.assertEqual(view.viewportTransform(), restored)


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for folder, ids in [('raw', (1, 2, 3)), ('log', (1, 3)), ('mask', (1, 2, 3))]:
            directory = self.root / folder
            directory.mkdir()
            for number in ids:
                Image.fromarray(np.arange(35, dtype=np.uint8).reshape(5, 7) + number).save(
                    directory / f'proj_{number}.png')
        self.window = ReviewWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.wait_jobs()
        self.temp.cleanup()

    def wait_jobs(self):
        deadline = time.monotonic() + 10
        while (self.window.jobs or any(p.orth_job or p._orth_requested for p in self.window.panes)) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertFalse(self.window.jobs, 'Import worker did not finish')
        self.assertFalse(any(p.orth_job or p._orth_requested for p in self.window.panes))

    def import_three(self):
        self.window.import_paths([self.root / name for name in ('raw', 'log', 'mask')])
        self.wait_jobs()
        return self.window.panes[:3]

    def zoom(self, view, ticks=1):
        pos = view.viewport().rect().center()
        event = QWheelEvent(QPointF(pos), QPointF(view.viewport().mapToGlobal(pos)),
            QPoint(), QPoint(0, 120 * ticks), Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier, Qt.ScrollPhase.NoScrollPhase, False)
        # Each event is a single step in the application's wheel handler.
        for _ in range(ticks):
            self.app.sendEvent(view.viewport(), event)

    def test_displayed_datasets_remain_highlighted_and_list_all_canvas_numbers(self):
        first, second, third = self.import_three()
        items = {self.window.dataset_list.item(i).data(Qt.ItemDataRole.UserRole):
                 self.window.dataset_list.item(i) for i in range(self.window.dataset_list.count())}
        self.assertEqual(items[first.dataset.id].data(DISPLAY_PANES_ROLE), [1])
        self.assertEqual(items[second.dataset.id].data(DISPLAY_PANES_ROLE), [2])
        displaced = third.dataset
        self.window.place(first.dataset, 2)
        self.assertEqual(items[first.dataset.id].data(DISPLAY_PANES_ROLE), [1, 3])
        self.assertEqual(items[displaced.id].data(DISPLAY_PANES_ROLE), [])
        first.clear()
        remaining_id = third.dataset.id
        self.assertEqual(items[remaining_id].data(DISPLAY_PANES_ROLE), [3])
        third.clear()
        self.assertEqual(items[displaced.id].data(DISPLAY_PANES_ROLE), [])
        self.assertEqual(items[remaining_id].data(DISPLAY_PANES_ROLE), [])

    def test_zoom_pan_and_markers_follow_binding_but_windows_stay_independent(self):
        first, second, third = self.import_three()
        self.zoom(first.view, 4)
        for pane in (second, third):
            self.assertAlmostEqual(pane.view.view_state()[0], first.view.view_state()[0], places=5)
        previous_width = second.ww.value()
        first.ww.setValue(13)
        first.drag_window(10, 5)
        self.assertEqual(second.ww.value(), previous_width)
        center = first.view.viewport().rect().center()
        QTest.mousePress(first.view.viewport(), Qt.MouseButton.MiddleButton, pos=center)
        QTest.mouseMove(first.view.viewport(), center + QPoint(20, 15))
        QTest.mouseRelease(first.view.viewport(), Qt.MouseButton.MiddleButton, pos=center + QPoint(20, 15))
        for pane in (second, third):
            np.testing.assert_allclose(pane.view.view_state()[1:3], first.view.view_state()[1:3], atol=.015)
        third.bind.setChecked(False)
        independent_state = third.view.view_state()
        self.zoom(first.view)
        self.assertEqual(third.view.view_state(), independent_state)
        position = first.view.mapFromScene(QPointF(3, 2))
        QTest.mouseClick(first.view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertEqual(first.view.crosshair, second.view.crosshair)
        self.assertIsNone(third.view.crosshair)
        self.window.link.setChecked(False)
        before = second.view.view_state()
        marks = second.view.crosshair
        self.zoom(first.view)
        position = first.view.mapFromScene(QPointF(4, 2))
        QTest.mouseClick(first.view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertEqual(second.view.view_state(), before)
        self.assertEqual(second.view.crosshair, marks)
        self.window.link.setChecked(True)
        third.bind.setChecked(True)
        self.assertAlmostEqual(third.view.view_state()[0], first.view.view_state()[0], places=5)
        self.assertEqual(third.view.crosshair, first.view.crosshair)

    def test_roi_auto_window_changes_only_the_clicked_canvas(self):
        first, second, _ = self.import_three()
        first._frame = np.full((100, 100), 60000., dtype=np.float32)
        first._frame[20:40, 30:50] = np.arange(400).reshape(20, 20)
        first.view.set_markers((.3, .2), (.3, .2, .5, .4))
        before = (second.ww.value(), second.wl.value())
        first.auto_button.click()
        expected_width, expected_level = auto_window(first._frame[20:40, 30:50])
        self.assertAlmostEqual(first.ww.value(), expected_width, places=5)
        self.assertAlmostEqual(first.wl.value(), expected_level, places=5)
        self.assertEqual((second.ww.value(), second.wl.value()), before)

    def test_slice_dropped_on_projection_canvas_is_routed_to_reconstruction(self):
        self.import_three()
        folder = self.root / 'slice'
        folder.mkdir()
        np.zeros((800, 800), dtype='<u2').tofile(folder / 'slice_0.slc')
        self.window.import_paths([folder], target=0)
        self.wait_jobs()
        recon = self.window.panes[3]
        self.assertEqual(recon.dataset.domain, 'reconstruction')
        self.assertEqual(recon._frame.shape, (800, 800))
        self.assertEqual(self.window.panes[0].dataset.domain, 'projection')

    def test_linked_layer_missing_frame_independent_windows_and_unlink(self):
        first, second, third = self.import_three()
        self.assertEqual([p.number for p in (first, second, third)], [1, 1, 1])
        original_width = second.ww.value()
        first.ww.setValue(12)
        first.slider.setValue(2)
        self.assertEqual([p.number for p in (first, second, third)], [2, 2, 2])
        self.assertIsNone(second._frame)
        self.assertFalse(second.view.has_image())
        self.assertEqual(second.ww.value(), original_width)
        self.assertEqual(first.ww.value(), 12)
        third.bind.setChecked(False)
        first.slider.setValue(3)
        self.assertEqual(third.number, 2)
        self.assertIsNotNone(second._frame)
        self.window.link.setChecked(False)
        first.slider.setValue(1)
        self.assertEqual(second.number, 3)
        self.window.link.setChecked(True)
        self.assertEqual(second.number, 1)

    def test_repeated_slider_scrubbing_and_windowing_do_not_shift_linked_images(self):
        panes = self.import_three()
        self.zoom(panes[0].view, 5)
        center = panes[0].view.viewport().rect().center()
        QTest.mousePress(panes[0].view.viewport(), Qt.MouseButton.MiddleButton, pos=center)
        QTest.mouseMove(panes[0].view.viewport(), center + QPoint(23, -17))
        QTest.mouseRelease(panes[0].view.viewport(), Qt.MouseButton.MiddleButton,
                           pos=center + QPoint(23, -17))
        transforms = [p.view.viewportTransform() for p in panes]
        for _ in range(100):
            panes[0].slider.setValue(2)  # The second dataset lacks this frame.
            panes[0].slider.setValue(3)
            panes[0].slider.setValue(1)
        for pane, before in zip(panes, transforms):
            self.assertEqual(pane.view.viewportTransform(), before)
        for _ in range(20):
            panes[0].ww.setValue(13)
            panes[0].wl.setValue(20)
            panes[0].auto_button.click()
        self.assertEqual(panes[0].view.viewportTransform(), transforms[0])

    def test_domain_guard_mask_mode_and_remove_detaches_all_views(self):
        first, _, _ = self.import_three()
        dataset = first.dataset
        self.window.place(dataset, 3)
        self.assertIsNone(self.window.panes[3].dataset)
        self.window.dataset_list.setCurrentRow(0)
        self.window.mask.setChecked(True)
        self.assertTrue(dataset.mask)
        self.assertFalse(first.ww.isEnabled())
        self.window.place(dataset, 2)
        self.window.remove_selected()
        self.assertIsNone(first.dataset)
        self.assertIsNone(self.window.panes[2].dataset)
        self.assertEqual(len(self.window.datasets), 2)

    def test_real_qt_file_drop_and_internal_dataset_drop(self):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.root / 'raw'))])
        viewport = self.window.panes[0].view.viewport()
        enter = QDragEnterEvent(QPoint(30, 30), Qt.DropAction.CopyAction, mime,
                               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, enter)
        self.assertTrue(enter.isAccepted())
        drop = QDropEvent(QPointF(30, 30), Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, drop)
        self.wait_jobs()
        self.assertEqual(len(self.window.datasets), 1)
        dataset = self.window.panes[0].dataset
        mime = QMimeData()
        mime.setData(MIME_DATASET, dataset.id.encode())
        viewport = self.window.panes[1].view.viewport()
        enter = QDragEnterEvent(QPoint(30, 30), Qt.DropAction.CopyAction, mime,
                               Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, enter)
        drop = QDropEvent(QPointF(30, 30), Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        self.app.sendEvent(viewport, drop)
        self.assertIs(self.window.panes[1].dataset, dataset)

    def test_clear_cancels_pending_import_and_releases_all_data(self):
        self.import_three()
        self.window.import_paths([self.root / 'raw'])
        self.window.clear_all()
        self.wait_jobs()
        self.assertEqual(self.window.datasets, {})
        self.assertEqual(self.window.dataset_list.count(), 0)
        self.assertTrue(all(p.dataset is None for p in self.window.panes))
        self.assertFalse(self.window.link_slider.isEnabled())

    def test_reconstruction_drop_stays_out_of_projection_binding(self):
        self.import_three()
        self.window.import_paths([self.root / 'raw'], target=3)
        self.wait_jobs()
        pane = self.window.panes[3]
        self.assertEqual(pane.dataset.domain, 'reconstruction')
        pane.slider.setValue(2)
        self.window.link_slider.setValue(3)
        self.assertEqual(pane.number, 2)

    def test_click_links_image_position_across_resolutions_and_zoom(self):
        first, second, third = self.import_three()
        path = self.root / 'larger.npy'
        np.save(path, np.zeros((4, 10, 14), dtype=np.uint8))
        self.window.import_paths([path])
        self.wait_jobs()
        larger = next(d for d in self.window.datasets.values() if d.paths == (path,))
        self.window.place(larger, 2)
        first.view.scale(1.5, 1.5)
        first.view._fit = False
        position = first.view.mapFromScene(QPointF(2, 2))
        QTest.mouseClick(first.view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertAlmostEqual(first.view.crosshair[0], 2 / 7, delta=.01)
        self.assertAlmostEqual(first.view.crosshair[1], 2 / 5, delta=.01)
        for pane in (second, third):
            self.assertEqual(pane.view.crosshair, first.view.crosshair)
            self.assertIsNone(pane.view.selection)
        self.assertIsNone(self.window.panes[3].view.crosshair)
        previous = first.view.crosshair
        first.view.fit_image()
        self.window.change_link_layer(2)  # Missing second-view layer hides its marks.
        self.assertFalse(second.view.has_image())
        self.window.change_link_layer(3)
        self.assertEqual(second.view.crosshair, previous)
        self.window.clear_all()
        self.assertTrue(all(p.view.crosshair is None for p in self.window.panes))

    def test_drag_rectangle_links_live_and_click_replaces_rectangle(self):
        panes = self.import_three()
        view = panes[0].view
        start = view.mapFromScene(QPointF(5, 4))
        end = view.mapFromScene(QPointF(2, 1))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        self.assertIsNotNone(view.selection)
        for pane in panes[1:]:
            self.assertEqual(pane.view.selection, view.selection)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        x0, y0, x1, y1 = view.selection
        self.assertAlmostEqual(x0, 2 / 7, delta=.01)
        self.assertAlmostEqual(y0, 1 / 5, delta=.01)
        self.assertAlmostEqual(x1, 5 / 7, delta=.01)
        self.assertAlmostEqual(y1, 4 / 5, delta=.01)
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertTrue(all(p.view.selection is None for p in panes))
        self.window.clear_markers()
        self.assertTrue(all(p.view.crosshair is None for p in panes))

    def test_selection_clamps_at_image_edges_and_middle_button_pans(self):
        first, _, _ = self.import_three()
        view = first.view
        start = view.mapFromScene(QPointF(2, 2))
        outside = QPoint(1, 1)
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), outside)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=outside)
        self.assertTrue(all(0 <= coordinate <= 1 for coordinate in view.selection))
        self.assertEqual(view.selection[:2], (0., 0.))
        markers = (view.crosshair, view.selection)
        view._fit = False
        view.scale(4, 4)
        before = view.horizontalScrollBar().value()
        center = view.viewport().rect().center()
        QTest.mousePress(view.viewport(), Qt.MouseButton.MiddleButton, pos=center)
        QTest.mouseMove(view.viewport(), center + QPoint(20, 10))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.MiddleButton, pos=center + QPoint(20, 10))
        self.assertNotEqual(view.horizontalScrollBar().value(), before)
        self.assertEqual((view.crosshair, view.selection), markers)

    def test_folder_drop_imports_root_and_subfolder_sequences_into_sidebar(self):
        Image.fromarray(np.zeros((5, 7), np.uint8)).save(self.root / 'overview.png')
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(self.root))])
        viewport = self.window.dataset_list.viewport()
        self.app.sendEvent(viewport, QDragEnterEvent(QPoint(20, 20), Qt.DropAction.CopyAction,
            mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        self.app.sendEvent(viewport, QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction,
            mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
        self.wait_jobs()
        self.assertEqual(self.window.dataset_list.count(), 4)
        self.assertEqual({d.paths[0].parent for d in self.window.datasets.values()},
                         {self.root, self.root / 'raw', self.root / 'log', self.root / 'mask'})


if __name__ == '__main__':
    unittest.main()
