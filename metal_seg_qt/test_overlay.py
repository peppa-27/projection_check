"""Projection overlays: source selection, exact colours, frame IDs and lifetimes."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton

from .app import DISPLAY_PANES_ROLE, ReviewWindow
from .data import Dataset, ImportOptions, display_pixels
from .overlay import composite_mask, mask_boundary


class OverlayPixelsTests(unittest.TestCase):
    def test_transparency_boundaries_and_raw_values_are_preserved(self):
        base = np.full((5, 6), 80, np.uint8)
        mask = np.zeros((5, 6), np.float32)
        mask[:3, :3] = 65535
        mask[4, 5] = np.nan
        original = mask.copy()
        rgb = composite_mask(base, mask, .5, (200, 40, 100))
        np.testing.assert_array_equal(rgb[1, 1], [140, 60, 90])
        np.testing.assert_array_equal(rgb[4, 5], [80, 80, 80])
        np.testing.assert_array_equal(mask, original)
        self.assertTrue(mask_boundary(mask == 65535)[0, 0])
        outline = composite_mask(base, mask, 1, (200, 40, 100), mode='outline')
        np.testing.assert_array_equal(outline[1, 1], [80, 80, 80])
        np.testing.assert_array_equal(outline[0, 0], [200, 40, 100])
        np.testing.assert_array_equal(composite_mask(base, mask, 0), np.repeat(base[..., None], 3, axis=2))
        with self.assertRaises(ValueError):
            composite_mask(base, np.zeros((3, 4)))

    def test_difference_colours_and_nonfinite_pixels(self):
        base = np.array([[1, 1, 0, 0, np.nan]])
        mask = np.array([[255, 0, 255, 0, np.inf]])
        pixels = display_pixels(base, 1, .5, True)
        actual = composite_mask(pixels, mask, 1, mode='difference', base=base)
        np.testing.assert_array_equal(actual[0],
            [[192, 203, 213], [255, 166, 69], [193, 117, 255], [0, 0, 0], [0, 0, 0]])
        missing = composite_mask(np.array([[255]], np.uint8), np.array([[np.nan]]),
                                 1, mode='difference', base=np.array([[1]]))
        np.testing.assert_array_equal(missing, [[[255, 255, 255]]])


class OverlayGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.errors = []
        self.old_hook = sys.excepthook
        sys.excepthook = lambda *args: self.errors.append(args)
        self.window = ReviewWindow()
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        try:
            self.window.clear_all()
            self.wait()
            self.window.close()
            self.app.processEvents()
            self.assertEqual(self.errors, [], 'Qt callback failed')
        finally:
            sys.excepthook = self.old_hook
            self.temp.cleanup()

    def wait(self):
        deadline = time.monotonic()+8
        while (self.window.jobs or any(p.orth_job or p._orth_requested for p in self.window.panes)) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertFalse(self.window.jobs)
        self.assertFalse(any(p.orth_job or p._orth_requested for p in self.window.panes))

    def source(self, name, frames, *, mask=False, domain='projection'):
        folder = self.root/name
        folder.mkdir()
        for number, frame in frames.items():
            Image.fromarray(frame).save(folder/f'frame_{number}.png')
        dataset = Dataset(list(folder.glob('*.png')), ImportOptions(mask=mask, domain=domain))
        self.window.add_generated_dataset(dataset)
        return dataset

    @staticmethod
    def color(view, x, y):
        return view.pixmap_item.pixmap().toImage().pixelColor(x, y).getRgb()[:3]

    def select_overlay(self, pane, source):
        index = pane.overlay_source.findData(source.id)
        self.assertGreater(index, 0)
        self.assertTrue(pane.overlay_source.isEnabled())
        pane.overlay_source.setCurrentIndex(index)

    def test_select_mask_over_raw_windowing_hide_and_zoom_stay_independent(self):
        raw = np.full((64, 80), 80, np.uint8)
        mask = np.zeros_like(raw)
        mask[10:30, 15:35] = 255
        base = self.source('raw', {5: raw})
        overlay = self.source('mask', {5: mask}, mask=True)
        self.window.place(base, 0)
        pane = self.window.panes[0]
        pane.ww.setValue(255)
        pane.wl.setValue(127.5)
        # Placing the base selects it in image management. The combo must use
        # its own selection, independently of that list's current item.
        self.assertIs(self.window.selected_dataset(), base)
        self.select_overlay(pane, overlay)
        self.app.processEvents()
        self.assertIs(pane.dataset, base)
        self.assertIs(pane.overlay_dataset, overlay)
        self.assertTrue(pane.overlay_panel.isVisible())
        self.assertEqual(self.color(pane.view, 20, 15), (168, 81, 87))
        self.assertEqual(self.color(pane.view, 0, 0), (80, 80, 80))
        pane.view._fit = False
        pane.view.scale(2.0736, 2.0736)
        pane.view.centerOn(30.25, 20.75)
        transform = pane.view.viewportTransform()
        pane.overlay_visible.setChecked(False)
        self.assertEqual(self.color(pane.view, 20, 15), (80, 80, 80))
        pane.overlay_visible.setChecked(True)
        pane.overlay_mode.setCurrentIndex(1)
        pane.overlay_opacity.setValue(100)
        self.assertEqual(self.color(pane.view, 20, 15), (80, 80, 80))
        self.assertEqual(self.color(pane.view, 15, 10), (255, 82, 94))
        pane.ww.setValue(160)
        pane.wl.setValue(80)
        self.assertEqual(self.color(pane.view, 20, 15), (128, 128, 128))
        self.assertEqual(pane.view.viewportTransform(), transform)
        pane.set_orthogonal_position((20.5/80, 15.5/64))
        self.assertIn('80', pane.pixel_value.text())
        self.assertIn('255', pane.pixel_value.text())
        np.testing.assert_array_equal(base.frame(5), raw)
        np.testing.assert_array_equal(overlay.frame(5), mask)
        item = next(self.window.dataset_list.item(row) for row in range(self.window.dataset_list.count())
                    if self.window.dataset_list.item(row).data(Qt.ItemDataRole.UserRole) == overlay.id)
        self.assertEqual(item.data(DISPLAY_PANES_ROLE), [1])
        self.assertIn('叠加', item.text())

    def test_real_ids_missing_frames_and_binding(self):
        a = np.full((8, 10), 40, np.uint8)
        b = np.zeros_like(a)
        b[2, 3] = 1
        base = self.source('raw', {5: a, 6: a, 7: a})
        overlay = self.source('mask', {5: b, 7: b}, mask=True)
        self.window.place(base, 0)
        self.window.place(base, 1)
        pane = self.window.panes[0]
        pane.ww.setValue(255)
        pane.wl.setValue(127.5)
        pane.set_overlay(overlay)
        pane.overlay_opacity.setValue(100)
        self.window.change_layer(0, 6)
        self.assertEqual(self.window.panes[1].number, 6)
        self.assertEqual(self.color(pane.view, 3, 2), (40, 40, 40))
        self.assertIn('缺少帧 6', pane.overlay_status.text())
        self.window.change_layer(0, 7)
        self.assertEqual(self.color(pane.view, 3, 2), (255, 82, 94))
        pane.set_layer(8)
        self.assertFalse(pane.view.has_image())
        self.assertIn('底图帧', pane.overlay_status.text())

    def test_mask_comparison_exact_categories_and_selection_values(self):
        a = np.zeros((8, 10), np.uint8)
        b = a.copy()
        a[2, 2:4] = 255
        b[2, 3:5] = 255
        base = self.source('base', {5: a}, mask=True)
        overlay = self.source('second', {5: b}, mask=True)
        self.window.place(base, 0)
        pane = self.window.panes[0]
        pane.set_overlay(overlay)
        pane.overlay_mode.setCurrentIndex(2)
        pane.overlay_opacity.setValue(100)
        self.assertEqual(self.color(pane.view, 2, 2), (255, 166, 69))
        self.assertEqual(self.color(pane.view, 3, 2), (192, 203, 213))
        self.assertEqual(self.color(pane.view, 4, 2), (193, 117, 255))
        self.assertIn('共有 1', pane.overlay_status.text())
        self.assertFalse(pane.overlay_color.isEnabled())

    def test_sections_align_actual_frame_ids_and_clear_rejects_late_results(self):
        a = np.full((8, 10), 40, np.uint8)
        b = np.zeros_like(a)
        b[2, 3] = 1
        base = self.source('raw', {5: a, 6: a, 7: a})
        overlay = self.source('mask', {5: b, 7: b, 9: b}, mask=True)
        self.window.place(base, 0)
        pane = self.window.panes[0]
        pane.set_overlay(overlay)
        pane.overlay_opacity.setValue(100)
        pane.view_mode.setCurrentIndex(0)
        pane.ww.setValue(255)
        pane.wl.setValue(127.5)
        pane.row_spin.setValue(2)
        pane.column_spin.setValue(3)
        self.wait()
        self.assertEqual(pane._overlay_sections[2], (5, 6, 7))
        for view, x in ((pane.row_view, 3), (pane.column_view, 2)):
            self.assertEqual(self.color(view, x, 0), (255, 82, 94))
            self.assertEqual(self.color(view, x, 1), (40, 40, 40))
            self.assertEqual(self.color(view, x, 2), (255, 82, 94))
        base.mask = True
        pane.overlay_mode.setCurrentIndex(2)
        # Unknown mask rows keep the base, instead of fabricating orange errors.
        self.assertEqual(self.color(pane.row_view, 3, 1), (255, 255, 255))
        base.mask = False
        pane.overlay_mode.setCurrentIndex(0)
        pane.row_spin.setValue(3)
        pane.clear_overlay()
        self.wait()
        self.assertIsNone(pane._overlay_sections)
        self.assertEqual(self.color(pane.column_view, 2, 0), (40, 40, 40))

    def test_invalid_sources_do_not_replace_overlay_and_remove_detaches(self):
        raw = np.zeros((8, 10), np.uint8)
        base = self.source('raw', {5: raw})
        good = self.source('good', {5: raw}, mask=True)
        self.window.place(base, 0)
        pane = self.window.panes[0]
        pane.set_overlay(good)
        for source in [base, self.source('wrong_size', {5: np.zeros((9, 10), np.uint8)}, mask=True),
                       self.source('wrong_ids', {9: raw}, mask=True),
                       self.source('reco', {5: raw}, mask=True, domain='reconstruction')]:
            self.assertFalse(self.window.place_overlay(source, 0))
            self.assertIs(pane.overlay_dataset, good)
        for row in range(self.window.dataset_list.count()):
            if self.window.dataset_list.item(row).data(Qt.ItemDataRole.UserRole) == good.id:
                self.window.dataset_list.setCurrentRow(row)
        self.window.remove_selected()
        self.assertIsNone(pane.overlay_dataset)
        self.assertIs(pane.dataset, base)
        self.assertNotIn(good.id, self.window.datasets)

    def test_sidebar_follows_active_canvas_and_does_not_shrink_images(self):
        raw = np.zeros((32, 40), np.uint8)
        base = self.source('raw', {5: raw})
        mask = self.source('mask', {5: raw}, mask=True)
        self.window.place(base, 0)
        self.window.place(base, 1)
        pane = self.window.panes[0]
        self.app.processEvents()
        size = pane.view.viewport().size()
        pane.overlay_toggle.click()
        self.app.processEvents()
        self.assertTrue(self.window.overlay_tools_panel.isVisible())
        self.assertEqual(pane.view.viewport().size(), size)
        self.select_overlay(pane, mask)
        pane.overlay_opacity.setValue(37)
        self.window.select_pane(1)
        self.assertEqual(self.window.overlay_stack.currentIndex(), 1)
        self.assertFalse(pane.overlay_panel.isVisible())
        self.assertIsNone(self.window.panes[1].overlay_source.currentData())
        self.window.select_pane(0)
        self.assertEqual(pane.overlay_source.currentData(), mask.id)
        self.assertEqual(pane.overlay_opacity.value(), 37)
        self.assertTrue(pane.overlay_panel.isVisible())
        self.window.overlay_tools_toggle.setChecked(False)
        self.assertIs(pane.overlay_dataset, mask)
        self.assertFalse(pane.overlay_toggle.isChecked())
        self.window.place(base, 0)
        self.assertIsNone(pane.overlay_dataset)
        self.assertIsNone(pane.overlay_source.currentData())
        self.assertIs(pane.dataset, base)

    def test_imported_mask_is_available_without_automatically_overlaying(self):
        a = np.full((8, 10), 40, np.uint8)
        base = self.source('raw', {5: a})
        self.window.place(base, 0)
        pane = self.window.panes[0]
        folder = self.root/'external'
        folder.mkdir()
        Image.fromarray((a != 0).astype(np.uint8)*255).save(folder/'mask_5.png')
        self.window.import_mask.setChecked(True)
        self.window.import_paths([folder])
        self.wait()
        self.assertIs(pane.dataset, base)
        self.assertIsNone(pane.overlay_dataset)
        overlay = next(d for d in self.window.datasets.values() if d is not base)
        self.select_overlay(pane, overlay)
        self.assertIs(pane.overlay_dataset, overlay)
        pane.overlay_remove.click()
        self.assertIsNone(pane.overlay_dataset)
        self.assertIsNone(pane.overlay_source.currentData())
        self.assertIn(overlay.id, self.window.datasets)

    def test_source_list_tracks_metadata_removal_and_clear(self):
        raw = np.zeros((8, 10), np.uint8)
        base = self.source('raw', {5: raw})
        candidate = self.source('candidate', {5: raw})
        reco = self.source('reco', {5: raw}, mask=True, domain='reconstruction')
        pane = self.window.panes[0]
        self.assertEqual(pane.overlay_source.count(), 1)
        self.assertFalse(pane.overlay_source.isEnabled())
        self.assertFalse(any(b.text() == '用所选'
                             for b in pane.overlay_panel.findChildren(QPushButton)))
        self.window.place(base, 0)
        self.window.dataset_list.setCurrentRow(1)
        self.window.mask.setChecked(True)
        self.assertEqual(pane.overlay_source.findData(reco.id), -1)
        self.select_overlay(pane, candidate)
        pane.overlay_opacity.setValue(37)
        another = self.source('another', {5: raw}, mask=True)
        self.assertEqual(pane.overlay_source.currentData(), candidate.id)
        self.assertEqual(pane.overlay_opacity.value(), 37)
        pane.overlay_source.setCurrentIndex(0)
        self.assertIsNone(pane.overlay_dataset)
        self.select_overlay(pane, candidate)
        self.window.dataset_list.setCurrentRow(1)
        self.window.mask.setChecked(False)
        self.assertIsNone(pane.overlay_dataset)
        self.assertEqual(pane.overlay_source.findData(candidate.id), -1)
        self.window.mask.setChecked(True)
        self.select_overlay(pane, candidate)
        self.window.dataset_list.setCurrentRow(1)
        self.window.domain.setCurrentIndex(1)
        self.assertIsNone(pane.overlay_dataset)
        self.assertEqual(pane.overlay_source.findData(candidate.id), -1)
        self.select_overlay(pane, another)
        self.window.dataset_list.setCurrentRow(3)
        self.window.remove_selected()
        self.assertIsNone(pane.overlay_dataset)
        self.assertEqual(pane.overlay_source.findData(another.id), -1)
        self.window.clear_all()
        self.assertEqual(pane.overlay_source.count(), 1)
        self.assertIsNone(pane.overlay_source.currentData())
        self.assertFalse(pane.overlay_source.isEnabled())

    def test_invalid_selection_restores_previous_source(self):
        raw = np.zeros((8, 10), np.uint8)
        base = self.source('raw', {5: raw})
        good = self.source('good', {5: raw}, mask=True)
        bad = self.source('wrong_size', {5: np.zeros((9, 10), np.uint8)}, mask=True)
        self.window.place(base, 0)
        pane = self.window.panes[0]
        self.select_overlay(pane, good)
        self.select_overlay(pane, bad)
        self.assertIs(pane.overlay_dataset, good)
        self.assertEqual(pane.overlay_source.currentData(), good.id)
        self.assertIn('尺寸不匹配', self.window.statusBar().currentMessage())


if __name__ == '__main__':
    unittest.main()
