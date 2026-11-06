"""Array-order and UI regressions for the three orthogonal inspection views."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch
import gc
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from .app import ReviewWindow
from .data import Dataset, ImportOptions, read_image
from .orthogonal import SequenceSpec, read_sections, section_indices


class ArraySectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Each axis has a different numeric contribution; flips/transposes
        # cannot accidentally pass by symmetry or constant test fixtures.
        n, y, x = np.indices((4, 5, 7))
        self.volume = (100 * n + 10 * y + x - 200).astype('<i2')

    def test_npy_and_raw_volume_sections_are_exact_and_keep_negative_values(self):
        for suffix in ('.npy', '.bin'):
            path = self.root / ('volume' + suffix)
            if suffix == '.npy':
                np.save(path, self.volume)
            else:
                self.volume.tofile(path)
            dataset = Dataset([path], ImportOptions(height=5, width=7, dtype='<i2', use_xml=False))
            spec = SequenceSpec.from_dataset(dataset)
            dataset.close()  # Background source must own its mappings.
            rows, columns, indices, errors = read_sections(spec, 2, 3, Event())
            self.assertFalse(errors)
            self.assertEqual(indices, (0, 1, 2, 3))
            np.testing.assert_array_equal(rows, self.volume[:, 2, :])
            np.testing.assert_array_equal(columns, self.volume[:, :, 3])
            path.unlink()  # Independent background mappings were closed too.

    def test_image_and_signed_slc_sequences_match_array_order_with_missing_layer_blank(self):
        for suffix in ('.tif', '.slc'):
            folder = self.root / suffix[1:]
            folder.mkdir()
            for index in (0, 1, 3):
                path = folder / f'slice_{index}{suffix}'
                if suffix == '.slc':
                    self.volume[index].tofile(path)
                else:
                    Image.fromarray(self.volume[index]).save(path)
            dataset = Dataset(list(folder.iterdir()), ImportOptions(
                reconstruction_height=5, reconstruction_width=7, auto_reconstruction_shape=False))
            spec = SequenceSpec.from_dataset(dataset)
            dataset.close()
            rows, columns, indices, errors = read_sections(spec, 1, 4, Event())
            self.assertFalse(errors)
            self.assertEqual(indices, (0, 1, 2, 3))
            np.testing.assert_array_equal(rows[[0, 1, 3]], self.volume[[0, 1, 3], 1, :])
            np.testing.assert_array_equal(columns[[0, 1, 3]], self.volume[[0, 1, 3], :, 4])
            self.assertTrue(np.isnan(rows[2]).all())

    def test_rgb_cancellation_and_bad_later_frame(self):
        folder = self.root / 'rgb'
        folder.mkdir()
        rgb = np.arange(3 * 5 * 7 * 3, dtype=np.uint8).reshape(3, 5, 7, 3)
        for n, image in enumerate(rgb):
            Image.fromarray(image).save(folder / f'frame_{n}.png')
        dataset = Dataset(list(folder.iterdir()), ImportOptions())
        spec = SequenceSpec.from_dataset(dataset)
        cached_spec = SequenceSpec.from_dataset(dataset, cache_sections=True)
        dataset.close()
        rows, columns, _, errors = read_sections(spec, 2, 3, Event())
        np.testing.assert_array_equal(rows, rgb[:, 2])
        np.testing.assert_array_equal(columns, rgb[:, :, 3])
        self.assertFalse(errors)
        cancelled = Event()
        cancelled.set()
        self.assertIsNone(read_sections(spec, 2, 3, cancelled))
        (folder / 'frame_1.png').write_bytes(b'broken image')
        rows, _, _, errors = read_sections(spec, 2, 3, Event())
        self.assertEqual(len(errors), 1)
        self.assertTrue((rows[1] == 0).all())
        cached_rows, _, _, cached_errors = read_sections(cached_spec, 2, 3, Event())
        self.assertEqual(len(cached_errors), 1)
        self.assertTrue((cached_rows[1] == 0).all())
        self.assertEqual(section_indices((2_000_000_000, 1)), (1, 2_000_000_000))

    def test_cached_raw_gray_and_rgb_sequences_are_read_once_and_keep_sparse_coordinates(self):
        rgb = np.arange(3 * 5 * 7 * 3, dtype=np.uint8).reshape(3, 5, 7, 3)
        for kind in ('raw', 'gray', 'rgb'):
            folder = self.root / kind
            folder.mkdir()
            values = rgb if kind == 'rgb' else self.volume[:3]
            for position, number in enumerate((2, 4, 5)):
                if kind == 'raw':
                    values[position].tofile(folder / f'slice_{number}.slc')
                else:
                    Image.fromarray(values[position]).save(folder / f'frame_{number}.tif')
            dataset = Dataset(list(folder.iterdir()), ImportOptions(
                reconstruction_height=5, reconstruction_width=7, auto_reconstruction_shape=False))
            spec = SequenceSpec.from_dataset(dataset, cache_sections=True)
            self.addCleanup(dataset.close)
            target = 'metal_seg_qt.orthogonal.np.fromfile' if kind == 'raw' else 'metal_seg_qt.orthogonal.read_image'
            reader = np.fromfile if kind == 'raw' else read_image
            with patch(target, wraps=reader) as reads:
                for row, column in ((1, 2), (3, 4), (2, 6)):
                    rows, columns, indices, errors = read_sections(spec, row, column, Event())
                    self.assertEqual(indices, (2, 3, 4, 5))
                    self.assertFalse(errors)
                    np.testing.assert_array_equal(rows[[0, 2, 3]], values[:, row])
                    np.testing.assert_array_equal(columns[[0, 2, 3]], values[:, :, column])
                    self.assertTrue((rows[1] == 0).all() if kind == 'rgb' else np.isnan(rows[1]).all())
                self.assertEqual(reads.call_count, 3)

    def test_cache_resume_after_cancel_and_reader_lifetime_after_dataset_close(self):
        folder = self.root / 'resume'
        folder.mkdir()
        for n, frame in enumerate(self.volume):
            Image.fromarray(frame).save(folder / f'frame_{n}.tif')
        dataset = Dataset(list(folder.iterdir()), ImportOptions())
        spec = SequenceSpec.from_dataset(dataset, cache_sections=True)
        second = SequenceSpec.from_dataset(dataset, cache_sections=True)
        self.assertIs(spec.volume_cache, second.volume_cache)
        cancelled = Event()

        def progress(done, total):
            if done == 1:
                cancelled.set()

        with patch('metal_seg_qt.orthogonal.read_image', wraps=read_image) as reads:
            self.assertIsNone(read_sections(spec, 1, 2, cancelled, progress))
            self.assertEqual(spec.volume_cache.completed, 1)
            path = spec.volume_cache.path
            dataset.close()
            rows, columns, _, errors = read_sections(second, 3, 4, Event())
            self.assertFalse(errors)
            np.testing.assert_array_equal(rows, self.volume[:, 3])
            np.testing.assert_array_equal(columns, self.volume[:, :, 4])
            self.assertEqual(reads.call_count, 4)  # Resume rather than re-read the first frame.
        del spec, second
        gc.collect()
        self.assertFalse(path.exists())

    def test_cache_falls_back_for_mixed_dtypes_and_disk_errors_without_truncation(self):
        folder = self.root / 'mixed'
        folder.mkdir()
        Image.fromarray(np.ones((5, 7), np.uint8)).save(folder / 'frame_0.tif')
        Image.fromarray(np.full((5, 7), 4000, np.uint16)).save(folder / 'frame_1.tif')
        for disk_failure in (False, True):
            dataset = Dataset(list(folder.iterdir()), ImportOptions())
            self.addCleanup(dataset.close)
            spec = SequenceSpec.from_dataset(dataset, cache_sections=True)
            if disk_failure:
                with patch('metal_seg_qt.orthogonal.np.lib.format.open_memmap', side_effect=OSError('disk full')):
                    rows, columns, _, errors = read_sections(spec, 1, 2, Event())
            else:
                rows, columns, _, errors = read_sections(spec, 1, 2, Event())
            self.assertFalse(errors)
            self.assertTrue((rows[0] == 1).all() and (rows[1] == 4000).all())
            self.assertTrue((columns[1] == 4000).all())
            self.assertTrue(spec.volume_cache.disabled)
            self.assertIsNone(spec.volume_cache.array)
            self.assertIsNone(spec.volume_cache.path)

    def test_simultaneous_readers_share_one_build_even_when_dataset_is_cleared(self):
        folder = self.root / 'shared'
        folder.mkdir()
        for n, frame in enumerate(self.volume):
            Image.fromarray(frame).save(folder / f'frame_{n}.tif')
        dataset = Dataset(list(folder.iterdir()), ImportOptions())
        first = SequenceSpec.from_dataset(dataset, cache_sections=True)
        second = SequenceSpec.from_dataset(dataset, cache_sections=True)
        started, release = Event(), Event()

        def progress(done, total):
            if done == 1:
                started.set()
                release.wait(2)

        with patch('metal_seg_qt.orthogonal.read_image', wraps=read_image) as reads:
            with ThreadPoolExecutor(max_workers=2) as pool:
                a = pool.submit(read_sections, first, 1, 2, Event(), progress)
                self.assertTrue(started.wait(2))
                b = pool.submit(read_sections, second, 3, 4, Event())
                dataset.close()
                release.set()
                rows_a, columns_a, _, errors_a = a.result(timeout=3)
                rows_b, columns_b, _, errors_b = b.result(timeout=3)
            self.assertEqual(reads.call_count, 4)
        self.assertFalse(errors_a or errors_b)
        np.testing.assert_array_equal(rows_a, self.volume[:, 1])
        np.testing.assert_array_equal(columns_a, self.volume[:, :, 2])
        np.testing.assert_array_equal(rows_b, self.volume[:, 3])
        np.testing.assert_array_equal(columns_b, self.volume[:, :, 4])


class ThreeViewWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        n, y, x = np.indices((4, 5, 7))
        self.volume = (100 * n + 10 * y + x).astype(np.float32)
        np.save(self.root / 'volume.npy', self.volume)
        self.window = ReviewWindow()
        self.window.show()
        self.app.processEvents()
        self.window.import_paths([self.root / 'volume.npy'])
        self.wait()

    def wait(self):
        deadline = time.monotonic() + 10
        while (self.window.jobs or any(p.orth_job or p._orth_requested for p in self.window.panes)) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertFalse(self.window.jobs)
        self.assertFalse(any(p.orth_job or p._orth_requested for p in self.window.panes))

    def tearDown(self):
        self.window.close()
        self.wait()
        self.temp.cleanup()

    def test_each_canvas_has_three_views_and_raw_sections_can_be_changed(self):
        pane = self.window.panes[0]
        self.assertTrue(all(len(p.section_views) == 3 and p.view_mode.currentIndex() == 1
                            for p in self.window.panes))
        self.assertTrue(pane.view.has_image())
        self.assertIsNone(pane._orth_data)
        self.assertFalse(pane.secondary_splitter.isVisible())
        pane.slider.setValue(2)
        self.assertTrue(pane.section_status.text().startswith('N=2'))
        pane.view_mode.setCurrentIndex(0)
        self.wait()
        self.assertTrue(all(v.has_image() for v in pane.section_views))
        pane.row_spin.setValue(1)
        pane.column_spin.setValue(4)
        self.wait()
        np.testing.assert_array_equal(pane._orth_data[0], self.volume[:, 1, :])
        np.testing.assert_array_equal(pane._orth_data[1], self.volume[:, :, 4])
        pane.view_mode.setCurrentIndex(2)
        self.assertTrue(pane.row_panel.isVisible())
        self.assertFalse(pane.primary_panel.isVisible())
        self.assertFalse(pane.column_panel.isVisible())
        pane.view_mode.setCurrentIndex(0)
        self.assertTrue(all(v.isVisible() for v in pane.section_views))

    def test_clicking_section_selects_layer_and_other_coordinate(self):
        pane = self.window.panes[0]
        pane.view_mode.setCurrentIndex(2)
        self.wait()
        position = pane.row_view.mapFromScene(QPointF(5.5, 2.5))
        QTest.mouseClick(pane.row_view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.wait()
        self.assertEqual(pane.number, 2)
        self.assertEqual(pane.column_spin.value(), 5)
        self.assertEqual(pane.row_spin.value(), 2)
        np.testing.assert_array_equal(pane._frame, self.volume[2])
        self.assertEqual(pane.pixel_value.text(), '值：225')
        self.assertIn('[N,Y,X]=[2,2,5]', pane.pixel_value.toolTip())

    def test_primary_pick_reads_raw_value_through_window_mask_and_layer_changes(self):
        pane = self.window.panes[0]
        self.assertEqual(pane.pixel_value.text(), '值：—')
        pane.slider.setValue(2)
        pane.ww.setValue(1)
        pane.wl.setValue(0)
        position = pane.view.mapFromScene(QPointF(3.5, 1.5))
        QTest.mouseClick(pane.view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.assertEqual(pane.pixel_value.text(), '值：213')
        self.assertIn('[N,Y,X]=[2,1,3]', pane.pixel_value.toolTip())
        pane.dataset.mask = True
        pane.render()
        self.assertEqual(pane.pixel_value.text(), '值：213')
        pane.slider.setValue(3)
        self.assertEqual(pane.pixel_value.text(), '值：313')
        self.window.clear_all()
        self.assertEqual(pane.pixel_value.text(), '值：—')

    def test_column_section_pick_reads_selected_voxel_and_drag_does_not_pick(self):
        pane = self.window.panes[0]
        start = pane.view.mapFromScene(QPointF(1.5, 1.5))
        end = pane.view.mapFromScene(QPointF(5.5, 3.5))
        QTest.mousePress(pane.view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(pane.view.viewport(), end)
        QTest.mouseRelease(pane.view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertIsNotNone(pane.view.selection)
        self.assertEqual(pane.pixel_value.text(), '值：—')
        pane.view_mode.setCurrentIndex(3)
        self.wait()
        position = pane.column_view.mapFromScene(QPointF(3.5, 1.5))
        QTest.mouseClick(pane.column_view.viewport(), Qt.MouseButton.LeftButton, pos=position)
        self.wait()
        self.assertEqual(pane.pixel_value.text(), '值：133')
        self.assertIn('[N,Y,X]=[1,3,3]', pane.pixel_value.toolTip())

    def test_new_requests_cancel_old_results_and_clear_during_read_is_safe(self):
        pane = self.window.panes[0]
        pane.view_mode.setCurrentIndex(0)
        self.wait()
        for index in range(5):
            pane.row_spin.setValue(index)
            self.app.processEvents()
        self.wait()
        np.testing.assert_array_equal(pane._orth_data[0], self.volume[:, 4, :])
        pane.column_spin.setValue(1)
        pane.start_sections()
        self.window.clear_all()
        self.wait()
        self.assertIsNone(pane.dataset)
        self.assertIsNone(pane._orth_data)
        self.assertTrue(all(not v.has_image() for v in pane.section_views))

    def test_changing_auxiliary_planes_reuses_source_mapping_and_keeps_image_until_ready(self):
        pane = self.window.panes[0]
        pane.view_mode.setCurrentIndex(0)
        self.wait()
        with patch('metal_seg_qt.orthogonal.np.load', wraps=np.load) as loads:
            for row in (1, 3, 4):
                pane.row_spin.setValue(row)
                self.assertTrue(pane.row_view.has_image())
                self.wait()
                np.testing.assert_array_equal(pane._orth_data[0], self.volume[:, row])
            pane.column_spin.setValue(5)
            self.wait()
            np.testing.assert_array_equal(pane._orth_data[1], self.volume[:, :, 5])
            pane.slider.setValue(2)
            self.assertEqual(loads.call_count, 0)

    def test_bound_projection_coordinate_changes_follow_and_unbound_stays_independent(self):
        first, second = self.window.panes[:2]
        self.window.place(first.dataset, 1)
        first.row_spin.setValue(1)
        first.column_spin.setValue(5)
        self.wait()
        self.assertEqual((second.row_spin.value(), second.column_spin.value()), (1, 5))
        second.bind.setChecked(False)
        first.row_spin.setValue(3)
        self.wait()
        self.assertEqual(second.row_spin.value(), 1)

    def test_maximize_restore_keeps_data_view_mode_and_sidebar(self):
        pane = self.window.panes[0]
        dataset = pane.dataset
        pane.view_mode.setCurrentIndex(0)
        self.wait()
        pane.slider.setValue(2)
        pane.ww.setValue(123)
        before_width = pane.width()
        pane.maximize_button.click()
        self.app.processEvents()
        self.assertEqual(self.window.maximized_pane, 0)
        self.assertEqual(self.window.board_grid.getItemPosition(self.window.board_grid.indexOf(pane)),
                         (0, 0, 2, 2))
        self.assertGreater(pane.width(), before_width)
        self.assertTrue(all(not p.isVisible() for p in self.window.panes[1:]))
        self.assertTrue(self.window.dataset_list.isVisible())
        self.assertEqual(pane.maximize_button.text(), '还原')
        pane.maximize_button.click()
        self.app.processEvents()
        self.assertIsNone(self.window.maximized_pane)
        self.assertTrue(all(p.isVisible() for p in self.window.panes))
        for index, item in enumerate(self.window.panes):
            self.assertEqual(self.window.board_grid.getItemPosition(self.window.board_grid.indexOf(item)),
                             (index // 2, index % 2, 1, 1))
        self.assertIs(pane.dataset, dataset)
        self.assertEqual((pane.number, pane.ww.value(), pane.view_mode.currentIndex()), (2, 123, 0))
        self.window.panes[3].maximize_button.click()
        self.app.processEvents()
        self.assertEqual(self.window.maximized_pane, 3)
        QTest.keyClick(self.window, Qt.Key.Key_Escape)
        self.app.processEvents()
        self.assertIsNone(self.window.maximized_pane)
        self.assertTrue(all(p.isVisible() for p in self.window.panes))


if __name__ == '__main__':
    unittest.main()
