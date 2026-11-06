"""Screen-safe startup with decorations, mixed-DPI coordinates and small screens."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from PySide6.QtCore import QMargins, QRect
from PySide6.QtWidgets import QApplication

from .app import ReviewWindow
from .window_geometry import startup_geometry


class StartupGeometryTests(unittest.TestCase):
    def test_client_and_frame_fit_work_area_even_with_negative_screen_origins(self):
        for area in (QRect(0, 0, 1920, 1032), QRect(-2560, 0, 1707, 1019),
                     QRect(1920, 1080, 1280, 680), QRect(0, 0, 960, 500)):
            with self.subTest(area=area.getRect()):
                margins = QMargins(8, 31, 8, 8)
                client, minimum = startup_geometry(area, margins)
                frame = client.marginsAdded(margins)
                self.assertTrue(area.contains(frame))
                self.assertLessEqual(minimum.width(), client.width())
                self.assertLessEqual(minimum.height(), client.height())
                self.assertLessEqual(client.width(), 1680)
                self.assertLessEqual(client.height(), 960)
                self.assertLessEqual(abs(frame.center().x()-area.center().x()), 1)
                self.assertLessEqual(abs(frame.center().y()-area.center().y()), 1)


class StartupWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_first_show_recovers_offscreen_oversized_window_and_keeps_content_accessible(self):
        window = ReviewWindow()
        self.addCleanup(window.close)
        window.resize(4000, 2400)
        window.move(9000, 9000)
        window.show()
        for _ in range(4):
            self.app.processEvents()
        area = window.screen().availableGeometry()
        self.assertTrue(area.contains(window.frameGeometry()),
                        (area.getRect(), window.frameGeometry().getRect()))
        # A small desktop must scroll the workspace instead of enlarging its
        # containing native window to the content's minimum layout width.
        scroll = window.workspace_scroll
        if scroll.widget().minimumSizeHint().width() > scroll.viewport().width():
            self.assertGreater(scroll.horizontalScrollBar().maximum(), 0)
            scroll.horizontalScrollBar().setValue(scroll.horizontalScrollBar().maximum())
            self.assertTrue(area.contains(window.frameGeometry()))

    def test_later_show_does_not_override_user_window_position_or_size(self):
        window = ReviewWindow()
        self.addCleanup(window.close)
        window.show()
        for _ in range(3):
            self.app.processEvents()
        window.resize(1440, 900)
        window.move(67, 91)
        self.app.processEvents()
        geometry = window.geometry()
        window.hide()
        window.show()
        for _ in range(3):
            self.app.processEvents()
        self.assertEqual(window.geometry(), geometry)


if __name__ == '__main__':
    unittest.main()
