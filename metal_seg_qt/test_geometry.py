"""Analytic camera geometry and actual Qt pick/angle-change regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
import tempfile
import time
import unittest

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from .app import ReviewWindow
from .data import Dataset, ImportOptions
from .geometry import GeometryError, ProjectionGeometry


def write_case(root):
    proj, slices = root / 'proj', root / 'slice'
    proj.mkdir(parents=True)
    slices.mkdir()
    config = dict(Input_Views_Number=2, Input_Views_Start_Count=20,
                  Sensor_Horizontal_Size=120, Sensor_Vertical_Size=100,
                  Volume_Horizontal_Size=7, Volume_Vertical_Size=3,
                  Volume_Translation_X=0, Volume_Translation_Y=0, Volume_Translation_Z=0)
    (proj / 'cbct_task.xml').write_text('<root>' + ''.join(
        f'<{key} v="{value}"/>' for key, value in config.items()) + '</root>', encoding='utf-8')
    cameras = np.array([[[10, 0, 0, 60], [0, 10, 0, 50], [0, 0, .1, 1]],
                        [[0, -10, 0, 60], [10, 0, 0, 50], [0, 0, .1, 1]]])
    np.savetxt(proj / 'PTM.dat', cameras.reshape(2, 12))
    (slices / 'ImageParam.ini').write_text(
        '[FileModule]\nBeginIndex=10\nEndIndex=12\n'
        '[RawImageInfo]\nWidth=7\nHeight=5\n'
        '[ImageModule]\nImagePosition_x=-1.5\nImagePosition_y=-1.5\nImagePosition_z=2\n'
        'CosRow_x=1\nCosRow_y=0\nCosRow_z=0\nCosColumn_x=0\nCosColumn_y=1\nCosColumn_z=0\n'
        'Scale_x=.5\nScale_y=.75\nScale_z=2\n', encoding='utf-16')
    for number in (20, 21):
        Image.fromarray(np.zeros((100, 120), np.uint8)).save(proj / f'proj_{number}.png')
    for number in (10, 11, 12):
        np.arange(35, dtype='<i2').reshape(5, 7).tofile(slices / f'slice_{number}.slc')
    return proj, slices


class GeometryMathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proj, self.slices = write_case(self.root)
        self.source = Dataset(list(self.slices.glob('*.slc')), ImportOptions())
        self.addCleanup(self.source.close)
        self.geometry = ProjectionGeometry.from_dataset(self.source)

    def test_voxel_centres_signed_slice_order_and_analytic_cameras(self):
        voxel, world = self.geometry.voxel((5.5 / 7, 3.5 / 5), 11)
        self.assertEqual(voxel, (5, 3, 11))
        np.testing.assert_allclose(world, [1, .75, 0])
        # Camera 0: u = 10X+60, v = 10Y+50 at Z=0.
        np.testing.assert_allclose(self.geometry.project(world, 20, (100, 120)),
                                   [70.5 / 120, 58 / 100])
        # Camera 1 independently rotates the X/Y directions by 90 degrees.
        np.testing.assert_allclose(self.geometry.project(world, 21, (100, 120)),
                                   [53 / 120, 60.5 / 100])
        self.assertEqual(self.geometry.voxel((0, 0), 10)[1][2], 2)
        self.assertEqual(self.geometry.voxel((0, 0), 12)[1][2], -2)
        self.assertEqual(self.geometry.project(world, 20, (50, 60)),
                         self.geometry.project(world, 20, (100, 120)))

    def test_fussen_native_frame_flips_x_z_only_and_preserves_both_mapping_directions(self):
        path = self.slices / 'ImageParam.ini'
        path.write_text(path.read_text(encoding='utf-16') +
                        '\n[EquipmentModule]\nManufacturer=Fussen\n', encoding='utf-16')
        native = ProjectionGeometry.from_dataset(self.source)
        np.testing.assert_array_equal(native.origin, [1.5, -1.5, -2])
        voxel, world = native.voxel((5.5 / 7, 3.5 / 5), 11)
        self.assertEqual(voxel, (5, 3, 11))
        np.testing.assert_allclose(world, [-1, .75, 0])
        self.assertEqual(native.voxel((0, 0), 10)[1][2], -2)
        self.assertEqual(native.voxel((0, 0), 12)[1][2], 2)
        point = native.project(world, 20, (100, 120))
        np.testing.assert_allclose(point, [50.5 / 120, 58 / 100])
        intersection, line = native.backproject(point, 20, (100, 120)).section(11)
        np.testing.assert_allclose(intersection, [5.5 / 7, 3.5 / 5])
        self.assertIsNone(line)
        self.assertIn('Fussen', native.coordinate_convention)
        # Generic calibration fixtures retain their original coordinate frame.
        np.testing.assert_allclose(self.geometry.voxel((5.5 / 7, 3.5 / 5), 11)[1], [1, .75, 0])

    def test_native_slice_order_compares_volume_center_in_the_same_frame(self):
        path = self.slices / 'ImageParam.ini'
        path.write_text(path.read_text(encoding='utf-16').replace('ImagePosition_z=2', 'ImagePosition_z=0') +
                        '\n[EquipmentModule]\nManufacturer=Fussen\n', encoding='utf-16')
        xml = self.proj / 'cbct_task.xml'
        xml.write_text(xml.read_text(encoding='utf-8').replace('Volume_Translation_Z v="0"',
                                                            'Volume_Translation_Z v="2"'), encoding='utf-8')
        native = ProjectionGeometry.from_dataset(self.source)
        self.assertEqual(native.voxel((0, 0), 10)[1][2], 0)
        self.assertEqual(native.voxel((0, 0), 11)[1][2], 2)
        self.assertEqual(native.voxel((0, 0), 12)[1][2], 4)

    def test_reported_native_case_camera_and_metal_coordinates_regression(self):
        # Minimal geometric fixture copied from the reported native case;
        # it contains no image or patient information and performs no fitting.
        root = self.root / 'native_regression'
        (root / 'proj').mkdir(parents=True)
        (root / 'slice').mkdir()
        config = dict(Input_Views_Number=1, Input_Views_Start_Count=321,
                      Sensor_Horizontal_Size=768, Sensor_Vertical_Size=768,
                      Volume_Horizontal_Size=600, Volume_Vertical_Size=400,
                      Volume_Translation_X=0, Volume_Translation_Y=0, Volume_Translation_Z=0)
        (root / 'proj' / 'cbct_task.xml').write_text('<root>' + ''.join(
            f'<{key} v="{value}"/>' for key, value in config.items()) + '</root>', encoding='utf-8')
        matrix = np.array([[-6.82835, 3.17572, .00514, 574.82715],
                           [.26619, .97485, 7.45423, 383.8952], [.0006, .00224, 0, 1.]])
        np.savetxt(root / 'proj' / 'PTM.dat', matrix.reshape(1, 12))
        (root / 'slice' / 'ImageParam.ini').write_text(
            '[FileModule]\nBeginIndex=0\nEndIndex=399\n[RawImageInfo]\nWidth=600\nHeight=600\n'
            '[ImageModule]\nImagePosition_x=-75\nImagePosition_y=-75\nImagePosition_z=49.875\n'
            'CosRow_x=1\nCosRow_y=0\nCosRow_z=0\nCosColumn_x=0\nCosColumn_y=1\nCosColumn_z=0\n'
            'Scale_x=.25\nScale_y=.25\nScale_z=.25\n[EquipmentModule]\nManufacturer=Fussen\n', encoding='utf-16')
        descriptor = SimpleNamespace(paths=(root / 'slice' / 'slice_0.slc',),
                                     shape=(600, 600), indices=(0, 399))
        native = ProjectionGeometry.from_dataset(descriptor)
        _, world = native.voxel((230.5 / 600, 152.5 / 600), 262)
        np.testing.assert_allclose(world, [17.5, -37, 15.625])
        pixel = np.array(native.project(world, 321, (768, 768))) * 768 - .5
        np.testing.assert_allclose(pixel, [364.2759939414846, 505.5479816627498], atol=1e-8)
        ray = native.backproject((365.5 / 768, 474.5 / 768), 321, (768, 768))
        self.assertEqual(ray.section(153), (None, None))
        depths = ray.local_endpoints[:, 2]
        self.assertGreater(depths.min(), 244)
        self.assertLess(depths.max(), 252)
        point, _ = ray.section(246)
        np.testing.assert_allclose(np.array(point) * 600 - .5,
                                   [238.31664270195967, 128.31942312280322], atol=1e-7)

    def test_invalid_frames_out_of_view_backside_and_cropped_aspect_are_rejected(self):
        for world, number, shape in [([0, 0, 0], 19, (100, 120)),
                                     ([100, 100, 0], 20, (100, 120)),
                                     ([0, 0, -20], 20, (100, 120)),
                                     ([0, 0, 0], 20, (100, 100))]:
            with self.subTest(world=world, number=number, shape=shape), self.assertRaises(GeometryError):
                self.geometry.project(world, number, shape)

    def test_backprojection_contains_world_point_and_roundtrips_every_sample(self):
        _, world = self.geometry.voxel((5.5 / 7, 3.5 / 5), 11)
        normalized = self.geometry.project(world, 20, (100, 120))
        ray = self.geometry.backproject(normalized, 20, (100, 120))
        a, b = ray.world_endpoints
        self.assertLess(np.linalg.norm(np.cross(world - a, b - a)), 1e-8)
        for t in np.linspace(0, 1, 11):
            np.testing.assert_allclose(self.geometry.project(a + t * (b - a), 20, (100, 120)),
                                       normalized, atol=1e-10)
        point, line = ray.section(11)
        np.testing.assert_allclose(point, [5.5 / 7, 3.5 / 5])
        self.assertIsNone(line)
        self.assertTrue(np.all(ray.local_endpoints >= -.5 - 1e-10))
        self.assertTrue(np.all(ray.local_endpoints <= [6.5, 4.5, 2.5]))
        np.testing.assert_allclose(self.geometry.backproject(normalized, 20, (50, 60)).local_endpoints,
                                   ray.local_endpoints)

    def test_coplanar_ray_is_a_line_and_parallel_other_slice_has_no_intersection(self):
        # Optical axis is +X from source (-10,0,0); principal point (60,50).
        self.geometry.matrices[0] = [[6, 10, 0, 60], [5, 0, 10, 50], [.1, 0, 0, 1]]
        ray = self.geometry.backproject((60.5 / 120, 50.5 / 100), 20, (100, 120))
        point, line = ray.section(11)
        self.assertIsNone(point)
        np.testing.assert_allclose(line, [[0, .5], [1, .5]])
        self.assertEqual(ray.section(10), (None, None))

    def test_backprojection_is_invariant_to_oblique_world_coordinate_frame(self):
        _, world = self.geometry.voxel((5.5 / 7, 3.5 / 5), 11)
        normalized = self.geometry.project(world, 20, (100, 120))
        angle = .7
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        translation = np.array([5., -3., 2.])
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = rotation, translation
        rotated = replace(self.geometry, matrices=self.geometry.matrices @ np.linalg.inv(transform),
            origin=rotation @ self.geometry.origin + translation,
            column_step=rotation @ self.geometry.column_step,
            row_step=rotation @ self.geometry.row_step, slice_step=rotation @ self.geometry.slice_step)
        np.testing.assert_allclose(rotated.backproject(normalized, 20, (100, 120)).local_endpoints,
            self.geometry.backproject(normalized, 20, (100, 120)).local_endpoints, atol=1e-9)

    def test_nonintersecting_invalid_or_infinite_source_ray_is_rejected(self):
        for point, number, shape in [((.99, .99), 20, (100, 120)),
                                      ((.5, .5), 19, (100, 120)),
                                      ((1.1, .5), 20, (100, 120)),
                                      ((.5, .5), 20, (100, 100))]:
            with self.subTest(point=point, number=number), self.assertRaises(GeometryError):
                self.geometry.backproject(point, number, shape)
        self.geometry.matrices[0, 2] = [0, 0, 0, 1]
        with self.assertRaisesRegex(GeometryError, '有限焦点'):
            self.geometry.backproject((.5, .5), 20, (100, 120))

    def test_inconsistent_slice_order_is_rejected(self):
        path = self.slices / 'ImageParam.ini'
        path.write_text(path.read_text(encoding='utf-16').replace('ImagePosition_z=2',
                                                                 'ImagePosition_z=100'), encoding='utf-16')
        with self.assertRaisesRegex(GeometryError, '无法可靠确定重建层序'):
            ProjectionGeometry.from_dataset(self.source)

    def test_missing_calibration_and_foreign_case_are_rejected(self):
        other = self.root / 'other'
        projection, _ = write_case(other)
        source = Dataset(list(projection.glob('*.png')), ImportOptions())
        self.addCleanup(source.close)
        with self.assertRaisesRegex(GeometryError, '不属于同一个几何目录'):
            self.geometry.check_projection_dataset(source)
        (self.slices / 'ImageParam.ini').unlink()
        with self.assertRaisesRegex(GeometryError, '缺少重建 ImageParam.ini'):
            ProjectionGeometry.from_dataset(self.source)


class GeometryWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.proj, self.slices = write_case(self.root)
        self.window = ReviewWindow()
        self.window.show()
        self.app.processEvents()
        self.window.import_paths([self.proj, self.slices])
        self.wait_jobs()
        projection = self.window.panes[0].dataset
        self.window.place(projection, 1)
        self.window.place(projection, 2)
        self.window.panes[3].slider.setValue(11)

    def wait_jobs(self):
        deadline = time.monotonic() + 10
        while (self.window.jobs or any(p.orth_job or p._orth_requested for p in self.window.panes)) and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(.005)
        self.app.processEvents()
        self.assertFalse(self.window.jobs)
        self.assertFalse(any(p.orth_job or p._orth_requested for p in self.window.panes))

    def tearDown(self):
        self.window.close()
        self.wait_jobs()
        self.temp.cleanup()

    def pick(self):
        view = self.window.panes[3].view
        pos = view.mapFromScene(QPointF(5.5, 3.5))
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=pos)
        self.assertIsNotNone(self.window.reference)

    def pick_ray(self):
        view = self.window.panes[0].view
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                         pos=view.mapFromScene(QPointF(70.5, 58)))
        self.assertIsNotNone(self.window.ray_reference)

    def test_click_reconstruction_and_update_each_projection_angle(self):
        self.pick()
        recon = self.window.panes[3]
        np.testing.assert_allclose(recon.view.reference_point, [5.5 / 7, 3.5 / 5])
        self.assertTrue(all(p.view.reference_point is not None for p in self.window.panes))
        before = self.window.panes[0].view.reference_point
        self.window.link.setChecked(False)
        self.window.panes[1].slider.setValue(21)
        np.testing.assert_allclose(self.window.panes[1].view.reference_point, [53 / 120, 60.5 / 100])
        self.assertEqual(self.window.panes[0].view.reference_point, before)
        transforms = [p.view.viewportTransform() for p in self.window.panes]
        for _ in range(30):
            self.window.panes[0].slider.setValue(21)
            self.window.panes[0].slider.setValue(20)
        for pane, transform in zip(self.window.panes, transforms):
            self.assertEqual(pane.view.viewportTransform(), transform)
        recon.slider.setValue(12)
        self.assertIsNone(recon.view.reference_point)
        self.assertEqual(self.window.panes[0].view.reference_point, before)
        recon.slider.setValue(11)
        self.assertIsNotNone(recon.view.reference_point)
        self.window.clear_markers()
        self.assertIsNone(self.window.reference)
        self.assertTrue(all(p.view.reference_point is None for p in self.window.panes))

    def test_switching_domains_does_not_leave_old_blue_or_ordinary_crosshairs(self):
        for pane in self.window.panes:
            pane.view.set_markers((.2, .3))
        self.pick()
        for pick in (self.pick_ray, self.pick, self.pick_ray):
            pick()
            for pane in self.window.panes:
                self.assertIsNone(pane.view.crosshair)
                self.assertIsNone(pane.view.navigation_point)
        self.window.change_link_layer(21)
        for pane in self.window.panes:
            self.assertIsNone(pane.view.crosshair)
            self.assertIsNone(pane.view.navigation_point)
        self.assertIsNotNone(self.window.ray_reference)

    def test_starting_rectangle_clears_previous_geometry_without_emitting_a_pick(self):
        self.pick()
        before = self.window.reference
        view = self.window.panes[0].view
        start, end = view.mapFromScene(QPointF(40, 40)), view.mapFromScene(QPointF(80, 70))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        self.assertIs(self.window.reference, before)
        QTest.mouseMove(view.viewport(), end)
        self.assertIsNone(self.window.reference)
        self.assertIsNone(self.window.ray_reference)
        self.assertTrue(all(p.view.reference_point is None and p.view.crosshair is None
                            for p in self.window.panes))
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertTrue(all(p.view.selection is not None for p in self.window.panes[:3]))
        self.assertIsNone(self.window.ray_reference)

    def test_projection_click_selects_ray_not_unique_3d_and_reconstruction_drag_remains_roi(self):
        view = self.window.panes[0].view
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                         pos=view.mapFromScene(QPointF(60, 50)))
        self.assertIsNone(self.window.reference)
        self.assertIsNotNone(self.window.ray_reference)
        view = self.window.panes[3].view
        start, end = view.mapFromScene(QPointF(1.5, 1.5)), view.mapFromScene(QPointF(4.5, 3.5))
        QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseMove(view.viewport(), end)
        QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=end)
        self.assertIsNotNone(view.selection)
        self.assertIsNone(self.window.reference)

    def test_ray_tracks_source_angle_and_current_slice_without_changing_view(self):
        self.pick_ray()
        recon = self.window.panes[3]
        self.assertIsNone(self.window.reference)
        self.assertIsNotNone(recon.view.ray_trace)
        self.assertIsNotNone(recon.view.reference_point)
        trace, point = recon.view.ray_trace, recon.view.reference_point
        transforms = [p.view.viewportTransform() for p in self.window.panes]
        recon.slider.setValue(12)
        self.assertEqual(recon.view.ray_trace, trace)
        self.assertNotEqual(recon.view.reference_point, point)
        self.window.link.setChecked(False)
        self.window.panes[1].slider.setValue(21)
        self.assertEqual(recon.view.ray_trace, trace)
        self.window.panes[0].slider.setValue(21)
        self.assertNotEqual(recon.view.ray_trace, trace)
        for _ in range(20):
            self.window.panes[0].slider.setValue(20)
            self.window.panes[0].slider.setValue(21)
        for pane, transform in zip(self.window.panes, transforms):
            self.assertEqual(pane.view.viewportTransform(), transform)

    def test_ray_and_point_modes_toggle_clear_and_detach(self):
        self.pick_ray()
        self.window.geometry_enabled.setChecked(False)
        self.assertTrue(all(p.view.ray_trace is None and p.view.reference_point is None
                            for p in self.window.panes))
        self.window.geometry_enabled.setChecked(True)
        self.assertIsNotNone(self.window.panes[3].view.ray_trace)
        self.pick()
        self.assertIsNone(self.window.ray_reference)
        self.assertIsNone(self.window.panes[3].view.ray_trace)
        self.pick_ray()
        self.assertIsNone(self.window.reference)
        self.window.clear_markers()
        self.assertIsNone(self.window.ray_reference)
        self.assertTrue(all(p.view.ray_trace is None for p in self.window.panes))
        self.pick_ray()
        self.window.panes[0].clear()
        self.assertIsNone(self.window.ray_reference)
        self.assertIsNone(self.window.panes[3].view.ray_trace)

    def test_missing_source_frame_and_nonintersecting_ray_hide_old_trace(self):
        self.pick_ray()
        self.window.change_layer(0, 22)
        self.assertIsNone(self.window.panes[3].view.ray_trace)
        self.assertIn('当前层缺失', self.window.geometry_info.text())
        self.window.change_layer(0, 20)
        self.assertIsNotNone(self.window.panes[3].view.ray_trace)
        view = self.window.panes[0].view
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                         pos=view.mapFromScene(QPointF(118, 98)))
        self.assertIsNone(self.window.ray_reference)
        self.assertIsNone(self.window.panes[3].view.ray_trace)
        self.assertIn('不穿过', self.window.geometry_info.text())

    def test_failed_pick_drops_old_reference_and_clear_detaches_source(self):
        self.pick()
        source = self.window.panes[3].dataset
        (self.slices / 'ImageParam.ini').unlink()
        self.window._geometry_cache.clear()
        view = self.window.panes[3].view
        QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                         pos=view.mapFromScene(QPointF(2.5, 2.5)))
        self.assertIsNone(self.window.reference)
        self.assertTrue(all(p.view.reference_point is None for p in self.window.panes))
        self.assertIn('缺少重建 ImageParam.ini', self.window.geometry_info.text())
        self.assertIs(self.window.panes[3].dataset, source)

    def test_enable_toggle_source_removal_and_foreign_projection(self):
        self.pick()
        self.window.geometry_enabled.setChecked(False)
        self.assertTrue(all(p.view.reference_point is None for p in self.window.panes))
        self.window.geometry_enabled.setChecked(True)
        self.assertTrue(all(p.view.reference_point is not None for p in self.window.panes))
        other, _ = write_case(self.root / 'foreign')
        self.window.import_paths([other], target=0)
        self.wait_jobs()
        self.assertIsNone(self.window.panes[0].view.reference_point)
        self.assertIsNotNone(self.window.panes[1].view.reference_point)
        self.assertIn('不属于同一个几何目录', self.window.geometry_info.text())
        self.window.panes[3].clear()
        self.assertIsNone(self.window.reference)
        self.assertTrue(all(p.view.reference_point is None for p in self.window.panes))


if __name__ == '__main__':
    unittest.main()
