from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from threading import Event

import numpy as np
from PySide6.QtCore import (QMimeData, QObject, QPoint, QPointF, QRectF, QSize, QRunnable,
                            QSignalBlocker, Qt, QThreadPool, QTimer, QElapsedTimer, Signal, QMargins)
from PySide6.QtGui import (QColor, QDrag, QImage, QPainter, QPen, QPixmap,
                           QShortcut, QKeySequence, QTransform, QCursor)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsPixmapItem,
    QGraphicsScene, QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QSlider, QSpinBox, QSplitter, QStyle, QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget,
    QScrollArea, QSizePolicy, QStackedWidget)

from .data import ImportOptions, auto_window, display_pixels, load_datasets
from .geometry import GeometryError, ProjectionGeometry
from .orthogonal import SequenceSpec, read_sections, section_indices
from .overlay import aligned_sections, composite_mask, foreground
from .window_geometry import startup_geometry
from .windowing import IntensityStatistics, collect_statistics
from .processing import ProcessingSource, ProcessingCancelled, binary_threshold

MIME_DATASET = 'application/x-metal-review-dataset'
DISPLAY_PANES_ROLE = int(Qt.ItemDataRole.UserRole) + 1


def local_paths(mime):
    return [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]


def has_drop(mime):
    return mime.hasFormat(MIME_DATASET) or bool(local_paths(mime))


def button(text, callback):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    return widget


class WorkspaceTab(QPushButton):
    dragHovered = Signal()
    dropped = Signal(object)

    def __init__(self, text):
        super().__init__(text)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if has_drop(event.mimeData()):
            self.dragHovered.emit()
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if has_drop(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if has_drop(event.mimeData()):
            self.dropped.emit(event.mimeData())
            event.acceptProposedAction()


class ImageView(QGraphicsView):
    dropped = Signal(object)
    activated = Signal()
    stepped = Signal(int)
    windowDragged = Signal(float, float)
    markersChanged = Signal(object, object)
    viewChanged = Signal(object)
    pointPicked = Signal(object)
    selectionStarted = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setAcceptDrops(True)
        self.setBackgroundBrush(QColor('#090e16'))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.hint = '拖入文件、图像序列文件夹\n或左侧列表中的数据'
        self._fit = True
        self._window_pos = None
        self._highlight = False
        # Normalized image coordinates keep marks aligned across resolutions.
        self.crosshair = None
        self.selection = None
        self.reference_point = None
        self.navigation_point = None
        self.ray_trace = None
        self.ray_section = None
        self._selection_origin = None
        self._selection_press = None
        self._selection_dragging = False
        self._selection_elapsed = QElapsedTimer()
        self._pan_pos = None
        self._pending_view_state = None
        self._view_state_pending = False

    def has_image(self):
        return self.pixmap_item.isVisible() and not self.pixmap_item.pixmap().isNull()

    def fit_scale(self):
        bounds = self.pixmap_item.boundingRect()
        return min(max(1, self.viewport().width() - 4) / bounds.width(),
                   max(1, self.viewport().height() - 4) / bounds.height())

    def view_state(self):
        if not self.has_image():
            return self._pending_view_state
        bounds = self.pixmap_item.boundingRect()
        # QRect.center() truncates half pixels. Keep the viewport center in
        # floating point so linked view updates do not feed that error back.
        inverse, _ = self.viewportTransform().inverted()
        center = inverse.map(QPointF(self.viewport().width() / 2,
                                     self.viewport().height() / 2))
        return (self.transform().m11() / self.fit_scale(),
                center.x() / bounds.width(), center.y() / bounds.height(), self._fit)

    def apply_view_state(self, state):
        if state is None:
            return
        self._pending_view_state = state
        zoom, cx, cy, self._fit = state
        if not self.has_image():
            self._view_state_pending = True
            return
        self._view_state_pending = False
        if self._fit:
            self.fit_image(notify=False)
            return
        bounds = self.pixmap_item.boundingRect()
        scale = self.fit_scale() * zoom
        self.setTransform(QTransform.fromScale(scale, scale))
        self.centerOn(cx * bounds.width(), cy * bounds.height())

    def notify_view_changed(self):
        state = self.view_state()
        self._pending_view_state = state
        if state is not None:
            self.viewChanged.emit(state)

    def set_markers(self, crosshair=None, selection=None):
        self.crosshair, self.selection = crosshair, selection
        self.viewport().update()

    def set_reference_point(self, point=None):
        self.reference_point = point
        self.viewport().update()

    def set_navigation_point(self, point=None):
        self.navigation_point = point
        self.viewport().update()

    def set_ray_trace(self, trace=None, section=None):
        self.ray_trace, self.ray_section = trace, section
        self.viewport().update()

    def image_position(self, position, clamp=False):
        if not self.has_image():
            return None
        bounds = self.pixmap_item.boundingRect()
        point = self.mapToScene(position.toPoint())
        if not clamp and not bounds.contains(point):
            return None
        return (max(0., min(1., (point.x() - bounds.left()) / bounds.width())),
                max(0., min(1., (point.y() - bounds.top()) / bounds.height())))

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.has_image():
            return
        bounds = self.pixmap_item.boundingRect()
        painter.save()
        painter.setClipRect(bounds)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(QColor('#65e2d3'), 1)
        pen.setCosmetic(True)  # Always one screen pixel, even when zoomed.
        if self.navigation_point is not None:
            pen.setColor(QColor('#72a9e8'))
            pen.setStyle(Qt.PenStyle.DotLine)
            painter.setPen(pen)
            x = bounds.left() + self.navigation_point[0] * bounds.width()
            y = bounds.top() + self.navigation_point[1] * bounds.height()
            painter.drawLine(QPointF(bounds.left(), y), QPointF(bounds.right(), y))
            painter.drawLine(QPointF(x, bounds.top()), QPointF(x, bounds.bottom()))
            pen.setColor(QColor('#65e2d3'))
            pen.setStyle(Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        if self.crosshair is not None:
            x = bounds.left() + self.crosshair[0] * bounds.width()
            y = bounds.top() + self.crosshair[1] * bounds.height()
            painter.drawLine(QPointF(bounds.left(), y), QPointF(bounds.right(), y))
            painter.drawLine(QPointF(x, bounds.top()), QPointF(x, bounds.bottom()))
        if self.selection is not None:
            x0, y0, x1, y1 = self.selection
            pen.setColor(QColor('#ffcc73'))
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(bounds.left() + x0 * bounds.width(),
                                    bounds.top() + y0 * bounds.height(),
                                    (x1 - x0) * bounds.width(),
                                    (y1 - y0) * bounds.height()))
        for line, color, style in ((self.ray_trace, '#ffb85c', Qt.PenStyle.DashLine),
                                   (self.ray_section, '#ff525e', Qt.PenStyle.SolidLine)):
            if line is not None:
                pen.setColor(QColor(color))
                pen.setStyle(style)
                painter.setPen(pen)
                a, b = (QPointF(bounds.left() + p[0] * bounds.width(),
                                bounds.top() + p[1] * bounds.height()) for p in line)
                painter.drawLine(a, b)
        if self.reference_point is not None:
            pen.setColor(QColor('#ff525e'))
            pen.setStyle(Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            x = bounds.left() + self.reference_point[0] * bounds.width()
            y = bounds.top() + self.reference_point[1] * bounds.height()
            painter.drawLine(QPointF(bounds.left(), y), QPointF(bounds.right(), y))
            painter.drawLine(QPointF(x, bounds.top()), QPointF(x, bounds.bottom()))
        painter.restore()

    def update_selection(self, position):
        if self._selection_origin is None:
            return
        delta = position - self._selection_press
        distance = delta.manhattanLength()
        long_press = self._selection_elapsed.elapsed() >= QApplication.startDragTime()
        if not self._selection_dragging and (distance >= QApplication.startDragDistance() or
                                             (long_press and distance >= 2)):
            self._selection_dragging = True
            self.selectionStarted.emit()
        if not self._selection_dragging:
            return
        end = self.image_position(position, clamp=True)
        if end is None:
            return
        x, y = self._selection_origin
        selection = (min(x, end[0]), min(y, end[1]), max(x, end[0]), max(y, end[1]))
        self.set_markers(None, selection)
        self.markersChanged.emit(self.crosshair, self.selection)

    def cancel_selection(self):
        self._selection_origin = None
        self._selection_press = None
        self._selection_dragging = False
        self._selection_elapsed.invalidate()

    def show_pixels(self, pixels, reset=False):
        previous = self.view_state()
        if pixels is None:
            self._pending_view_state = previous
            # A missing layer is a display gap, not a geometry change. Keeping
            # the bounds also keeps scroll ranges and exact pixel positions.
            self.pixmap_item.setVisible(False)
            if reset:
                self.cancel_selection()
                self.pixmap_item.setPixmap(QPixmap())
                self.scene().setSceneRect(0, 0, 1, 1)
                self._pending_view_state = None
                self._view_state_pending = False
                self._fit = True
            self.viewport().update()
            return
        pixels = np.ascontiguousarray(pixels)
        height, width = pixels.shape[:2]
        fmt = (QImage.Format.Format_Grayscale8 if pixels.ndim == 2 else
               QImage.Format.Format_RGBA8888 if pixels.shape[2] == 4 else QImage.Format.Format_RGB888)
        image = QImage(pixels.data, width, height, pixels.strides[0], fmt).copy()
        bounds_before = self.pixmap_item.boundingRect()
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.pixmap_item.setVisible(True)
        geometry_changed = self.pixmap_item.boundingRect() != bounds_before
        if geometry_changed:
            self.scene().setSceneRect(self.pixmap_item.boundingRect())
        if reset:
            self._fit = True
            self._pending_view_state = None
            self._view_state_pending = False
        # Ordinary slice changes and windowing must leave both the transform
        # and integer scroll offsets untouched. Recentring on every redraw
        # accumulates the rounding inherent in QGraphicsView.centerOn().
        if not (reset or geometry_changed or self._view_state_pending):
            self.viewport().update()
            return
        if self._fit:
            self._view_state_pending = False
            self.fit_image(notify=False)
        elif previous is not None:
            self.apply_view_state(previous)

    def fit_image(self, notify=True):
        self._fit = True
        if self.has_image():
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            if notify:
                self.notify_view_changed()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit:
            self.fit_image(notify=False)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.has_image() or self._highlight:
            painter = QPainter(self.viewport())
            painter.setPen(QColor('#7f93ad'))
            if not self.has_image():
                painter.drawText(self.viewport().rect().adjusted(20, 20, -20, -20),
                    Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, self.hint)
            if self._highlight:
                painter.setPen(QPen(QColor('#55c9c0'), 3, Qt.PenStyle.DashLine))
                painter.drawRect(self.viewport().rect().adjusted(5, 5, -5, -5))
            painter.end()

    def dragEnterEvent(self, event):
        if has_drop(event.mimeData()):
            event.acceptProposedAction()
            self._highlight = True
            self.viewport().update()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if has_drop(event.mimeData()):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._highlight = False
        self.viewport().update()
        event.accept()

    def dropEvent(self, event):
        self._highlight = False
        self.viewport().update()
        if has_drop(event.mimeData()):
            self.activated.emit()
            self.dropped.emit(event.mimeData())
            event.acceptProposedAction()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            return
        self.activated.emit()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            current = self.transform().m11()
            factor = 1.2 if delta > 0 else 1 / 1.2
            if .01 <= current * factor <= 100:
                self._fit = False
                self.scale(factor, factor)
                self.notify_view_changed()
        else:
            self.stepped.emit(1 if delta > 0 else -1)
        event.accept()

    def mousePressEvent(self, event):
        self.activated.emit()
        if event.button() == Qt.MouseButton.LeftButton:
            self.cancel_selection()
            point = self.image_position(event.position())
            if point is not None:
                self._selection_origin = point
                self._selection_press = event.position()
                self._selection_dragging = False
                self._selection_elapsed.start()
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_pos = event.position()
            self._fit = False
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._window_pos = event.position()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selection_origin is not None:
            self.update_selection(event.position())
            event.accept()
        elif self._pan_pos is not None:
            delta = event.position() - self._pan_pos
            self._pan_pos = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - round(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - round(delta.y()))
            self.notify_view_changed()
            event.accept()
        elif self._window_pos is not None:
            delta = event.position() - self._window_pos
            self._window_pos = event.position()
            self.windowDragged.emit(delta.x(), delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.update_selection(event.position())
            if (self._selection_origin is not None and not self._selection_dragging and
                    self._selection_elapsed.elapsed() < QApplication.startDragTime()):
                self.set_markers(self._selection_origin)
                self.markersChanged.emit(self.crosshair, None)
                self.pointPicked.emit(self._selection_origin)
            self.cancel_selection()
            event.accept()
        elif event.button() == Qt.MouseButton.MiddleButton:
            self._pan_pos = None
            self.viewport().unsetCursor()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._window_pos = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.cancel_selection()
        self.fit_image()
        event.accept()


class SectionSignals(QObject):
    finished = Signal(int, object, str)
    progress = Signal(int, int, int)


class SectionJob(QRunnable):
    def __init__(self, token, spec, row, column, overlay_spec=None):
        super().__init__()
        self.token, self.spec, self.row, self.column = token, spec, row, column
        self.cancelled = Event()
        self.signals = SectionSignals()
        self.overlay_spec = overlay_spec
        self.overlay_result = None
        self.overlay_error = ''

    def run(self):
        result, error = None, ''
        try:
            result = read_sections(self.spec, self.row, self.column, self.cancelled,
                                  lambda done, total: self.signals.progress.emit(self.token, done, total))
            if result is not None and self.overlay_spec is not None and not self.cancelled.is_set():
                try:
                    overlay = read_sections(self.overlay_spec, self.row, self.column, self.cancelled)
                    if overlay is not None:
                        self.overlay_result = aligned_sections(overlay, result[2])
                except Exception as caught:
                    self.overlay_error = str(caught)
        except Exception as caught:
            error = str(caught)
        finally:
            self.spec = None
            self.overlay_spec = None
        self.signals.finished.emit(self.token, result, error)


class SectionPanel(QWidget):
    def __init__(self, title, coordinate=None):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        header = QHBoxLayout()
        header.setSpacing(3)
        self.label = QLabel(title)
        self.label.setObjectName('muted')
        header.addWidget(self.label, 1)
        self.coordinate = None
        if coordinate is not None:
            self.coordinate = QSpinBox()
            self.coordinate.setPrefix(coordinate + '=')
            self.coordinate.setKeyboardTracking(False)
            self.coordinate.setMaximumWidth(88)
            header.addWidget(self.coordinate)
        layout.addLayout(header)
        self.view = ImageView()
        self.view.setMinimumSize(90, 65)
        layout.addWidget(self.view, 1)


class ReviewPane(QFrame):
    layerChanged = Signal(int, int)
    chosen = Signal(int)
    dropReceived = Signal(int, object)
    bindingChanged = Signal()
    orthogonalChanged = Signal(int, object)
    auxiliaryViewChanged = Signal(int, int, object)
    maximizeRequested = Signal(int)
    overlaySourceChanged = Signal(int, object)
    overlayToolsRequested = Signal(int, bool)

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.dataset = None
        self.number = 0
        self._frame = None
        self.orth_job = None
        self._orth_token = 0
        self._orth_requested = None
        self._orth_data = None
        self._orth_signature = None
        self._active_plane = 0
        self._has_pixel_pick = False
        self.threshold_preview = None
        self.overlay_dataset = None
        self._overlay_frame = None
        self._overlay_sections = None
        self._overlay_error = ''
        self._overlay_revision = 0
        self._orth_timer = QTimer(self)
        self._orth_timer.setSingleShot(True)
        self._orth_timer.setInterval(20)
        self._orth_timer.timeout.connect(self.start_sections)
        self.setObjectName('pane')
        self.setProperty('active', index == 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        tag = QLabel(f'0{index + 1}  ' + ('投影域' if index < 3 else '重建域'))
        tag.setObjectName('paneTag')
        header.addWidget(tag)
        self.title = QLabel('尚未放置数据')
        self.title.setMinimumWidth(30)
        header.addWidget(self.title, 1)
        self.view_mode = QComboBox()
        self.view_mode.addItems(['三视图', '原始切片', '固定行截面', '固定列截面'])
        self.view_mode.setMaximumWidth(108)
        self.view_mode.setToolTip('按数组 [N,Y,X] 显示；可单独放大任一方向')
        self.view_mode.currentIndexChanged.connect(self.change_view_mode)
        header.addWidget(self.view_mode)
        self.overlay_toggle = QToolButton()
        self.overlay_toggle.setText('叠加')
        self.overlay_toggle.setCheckable(True)
        self.overlay_toggle.setVisible(index < 3)
        self.overlay_toggle.setToolTip('打开右侧此幕布的投影叠加设置；实心圆表示已添加叠加层')
        header.addWidget(self.overlay_toggle)
        self.maximize_button = button('放大', lambda: self.maximizeRequested.emit(index))
        self.maximize_button.setCheckable(True)
        self.maximize_button.setFixedWidth(48)
        self.maximize_button.setToolTip('占满中间四个画布区域；再次点击还原 2×2 布局（Esc 也可还原）')
        header.addWidget(self.maximize_button)
        self.bind = QCheckBox('联动')
        self.bind.setToolTip('参与投影的切层、标记、缩放和平移联动')
        self.bind.setChecked(index < 3)
        self.bind.setVisible(index < 3)
        self.bind.toggled.connect(self.bindingChanged)
        header.addWidget(self.bind)
        clear = button('×', self.clear)
        clear.setFixedWidth(26)
        clear.setToolTip('仅清空此画布，数据仍保留在左侧图像管理')
        header.addWidget(clear)
        layout.addLayout(header)
        self.overlay_panel = QWidget(self)
        overlay_layout = QVBoxLayout(self.overlay_panel)
        overlay_layout.setContentsMargins(0, 0, 0, 0)
        overlay_layout.setSpacing(4)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel('叠加图像'))
        self.overlay_source = QComboBox()
        self.overlay_source.setMinimumWidth(0)
        self.overlay_source.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.overlay_source.addItem('暂无投影掩膜', None)
        self.overlay_source.currentIndexChanged.connect(
            lambda _: self.overlaySourceChanged.emit(index, self.overlay_source.currentData()))
        source_row.addWidget(self.overlay_source, 1)
        self.overlay_remove = button('移除', self.clear_overlay)
        source_row.addWidget(self.overlay_remove)
        overlay_layout.addLayout(source_row)
        control_row = QHBoxLayout()
        self.overlay_visible = QCheckBox('显示')
        self.overlay_visible.setChecked(True)
        self.overlay_visible.toggled.connect(self.overlay_settings_changed)
        control_row.addWidget(self.overlay_visible)
        self.overlay_mode = QComboBox()
        for label, mode in [('彩色填充', 'fill'), ('掩膜轮廓', 'outline'), ('掩膜差异', 'difference')]:
            self.overlay_mode.addItem(label, mode)
        self.overlay_mode.setToolTip('差异模式：浅灰=共有，橙色=仅底图，紫色=仅叠加；颜色不代表对错')
        self.overlay_mode.currentIndexChanged.connect(self.overlay_settings_changed)
        control_row.addWidget(self.overlay_mode)
        self.overlay_color = QComboBox()
        for label, color in [('红色', (255, 82, 94)), ('青色', (65, 220, 210)),
                             ('黄色', (255, 211, 80)), ('紫色', (193, 117, 255))]:
            self.overlay_color.addItem(label, color)
        self.overlay_color.currentIndexChanged.connect(self.overlay_settings_changed)
        control_row.addWidget(self.overlay_color)
        overlay_layout.addLayout(control_row)
        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel('不透明度'))
        self.overlay_opacity = QSlider(Qt.Orientation.Horizontal)
        self.overlay_opacity.setRange(0, 100)
        self.overlay_opacity.setValue(50)
        self.overlay_opacity.setMinimumWidth(35)
        self.overlay_opacity.setToolTip('叠加不透明度；0% 只看底图，100% 显示完整掩膜颜色')
        self.overlay_opacity.valueChanged.connect(self.overlay_settings_changed)
        opacity_row.addWidget(self.overlay_opacity, 1)
        self.overlay_percent = QLabel('50%')
        self.overlay_percent.setFixedWidth(34)
        opacity_row.addWidget(self.overlay_percent)
        overlay_layout.addLayout(opacity_row)
        self.overlay_status = QLabel('按真实帧号叠加；底图和掩膜需具有相同尺寸与方向')
        self.overlay_status.setObjectName('muted')
        self.overlay_status.setWordWrap(True)
        self.overlay_status.setMinimumWidth(0)
        overlay_layout.addWidget(self.overlay_status)
        self.overlay_panel.hide()
        self.overlay_toggle.toggled.connect(lambda checked: self.overlayToolsRequested.emit(index, checked))
        self.primary_panel = SectionPanel('A · X→ / Y↓')
        self.row_panel = SectionPanel('B · X→ / N↓', 'Y')
        self.column_panel = SectionPanel('C · Y→ / N↓', 'X')
        self.row_spin = self.row_panel.coordinate
        self.column_spin = self.column_panel.coordinate
        self.row_spin.valueChanged.connect(self.coordinate_changed)
        self.column_spin.valueChanged.connect(self.coordinate_changed)
        self.view = self.primary_panel.view
        self.row_view, self.column_view = self.row_panel.view, self.column_panel.view
        self.section_views = [self.view, self.row_view, self.column_view]
        self.secondary_splitter = QSplitter(Qt.Orientation.Vertical)
        self.secondary_splitter.addWidget(self.row_panel)
        self.secondary_splitter.addWidget(self.column_panel)
        self.secondary_splitter.setChildrenCollapsible(False)
        self.image_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.image_splitter.addWidget(self.primary_panel)
        self.image_splitter.addWidget(self.secondary_splitter)
        self.image_splitter.setChildrenCollapsible(False)
        self.image_splitter.setSizes([300, 260])
        self.secondary_splitter.setSizes([130, 130])
        self.view.activated.connect(lambda: self.activate_plane(0))
        self.view.dropped.connect(lambda mime: self.dropReceived.emit(index, mime))
        self.view.stepped.connect(self.step)
        self.view.windowDragged.connect(self.drag_window)
        self.view.pointPicked.connect(self.pick_primary_position)
        for axis, view in enumerate((self.row_view, self.column_view), 1):
            view.activated.connect(lambda a=axis: self.activate_plane(a))
            view.dropped.connect(lambda mime: self.dropReceived.emit(index, mime))
            view.windowDragged.connect(self.drag_window)
            view.pointPicked.connect(lambda point, a=axis: self.pick_section_position(a, point))
            view.stepped.connect(lambda step, a=axis: self.step_section(a, step))
            view.viewChanged.connect(lambda state, a=axis: self.auxiliaryViewChanged.emit(index, a, state))
        layout.addWidget(self.image_splitter, 1)
        self.section_status = QLabel('数组轴：N（层/视角）、Y（行）、X（列）')
        self.section_status.setObjectName('muted')
        self.section_status.setMinimumWidth(0)
        self.pixel_value = QLabel('值：—')
        self.pixel_value.setObjectName('muted')
        self.pixel_value.setMaximumWidth(240)
        self.pixel_value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        status_row = QHBoxLayout()
        status_row.addWidget(self.section_status, 1)
        status_row.addWidget(self.pixel_value)
        layout.addLayout(status_row)
        row = QHBoxLayout()
        self.layer = QSpinBox()
        self.layer.setPrefix('层 ')
        self.layer.setRange(0, 0)
        self.layer.setKeyboardTracking(False)
        self.layer.valueChanged.connect(self.request_layer)
        row.addWidget(self.layer)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.request_layer)
        row.addWidget(self.slider, 1)
        self.count = QLabel('—')
        row.addWidget(self.count)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel('窗宽'))
        self.ww = QDoubleSpinBox()
        self.ww.setDecimals(6)
        self.ww.setMinimumWidth(0)
        self.ww.setRange(.000001, 1e15)
        self.ww.setValue(255)
        self.ww.setKeyboardTracking(False)
        self.ww.valueChanged.connect(self.render)
        row.addWidget(self.ww, 1)
        row.addWidget(QLabel('窗位'))
        self.wl = QDoubleSpinBox()
        self.wl.setDecimals(6)
        self.wl.setMinimumWidth(0)
        self.wl.setRange(-1e15, 1e15)
        self.wl.setValue(127.5)
        self.wl.setKeyboardTracking(False)
        self.wl.valueChanged.connect(self.render)
        row.addWidget(self.wl, 1)
        self.auto_button = button('自动调窗', self.reset_window)
        self.auto_button.setToolTip('独立调窗：有矩形时按框内像素的 1%–99% 灰度范围计算，否则使用整层；不改变缩放')
        row.addWidget(self.auto_button)
        fit_button = button('适应窗口', lambda: self.section_views[self._active_plane].fit_image())
        fit_button.setToolTip('调整缩放和平移，让整张图放进画布；不改变窗宽窗位')
        row.addWidget(fit_button)
        layout.addLayout(row)
        self.info = QLabel('左键定位/框选 · 中键平移 · 右键调窗')
        self.info.setObjectName('muted')
        layout.addWidget(self.info)
        self.clear()
        self.view_mode.setCurrentIndex(1)

    def set_overlay_sources(self, datasets):
        with QSignalBlocker(self.overlay_source):
            self.overlay_source.clear()
            self.overlay_source.addItem('不叠加（请选择投影掩膜）', None)
            for dataset in datasets:
                if dataset.domain == 'projection' and dataset.mask and len(dataset.shape) == 2:
                    self.overlay_source.addItem(dataset.name, dataset.id)
                    self.overlay_source.setItemData(self.overlay_source.count()-1,
                        str(dataset.paths[0]), Qt.ItemDataRole.ToolTipRole)
            if self.overlay_source.count() == 1:
                self.overlay_source.setItemText(0, '暂无投影掩膜')
        self.sync_overlay_source()
        self.update_overlay_controls()

    def sync_overlay_source(self):
        dataset = self.overlay_dataset
        with QSignalBlocker(self.overlay_source):
            self.overlay_source.setCurrentIndex(max(0,
                self.overlay_source.findData(dataset.id if dataset is not None else None)))
        self.overlay_source.setToolTip(
            f'叠加掩膜：{dataset.name}\n{dataset.paths[0]}\n按真实帧号匹配，不使用邻层代替'
            if dataset is not None else
            '直接选择图像管理中的投影域二值掩膜；新数据请先导入图像管理，并勾选「二值掩膜」')

    def set_overlay(self, dataset):
        if self.index >= 3 or self.dataset is None:
            raise ValueError('请先在投影幕布放置底图，再添加叠加掩膜')
        if dataset.domain != 'projection' or not dataset.mask or len(dataset.shape) != 2:
            raise ValueError('叠加层需要投影域二值掩膜；请在图像管理中检查数据域和「二值掩膜」')
        if tuple(dataset.shape) != tuple(self.dataset.shape[:2]):
            raise ValueError(f'尺寸不匹配：底图 {self.dataset.shape[:2]}，掩膜 {dataset.shape}；未自动缩放或配准')
        if not set(dataset.indices).intersection(self.dataset.indices):
            raise ValueError('底图与掩膜没有相同的真实帧号；请检查文件编号')
        self.overlay_dataset = dataset
        self.sync_overlay_source()
        with QSignalBlocker(self.overlay_visible):
            self.overlay_visible.setChecked(True)
        self.overlay_toggle.setChecked(True)
        self.load_overlay_frame()
        self.request_sections()
        self.render()
        self.update_pixel_value()
        self.bindingChanged.emit()

    def clear_overlay(self, *_, render=True):
        self.overlay_dataset = None
        self._overlay_frame = None
        self._overlay_sections = None
        self._overlay_error = ''
        self._overlay_revision += 1
        self.sync_overlay_source()
        if render:
            self.request_sections()
            self.render()
            self.update_pixel_value()
            self.bindingChanged.emit()

    def load_overlay_frame(self):
        self._overlay_frame = None
        self._overlay_error = ''
        if self.overlay_dataset is not None:
            try:
                frame = self.overlay_dataset.frame(self.number)
                if frame.ndim != 2 or frame.shape != self.dataset.shape[:2]:
                    raise ValueError('当前掩膜帧与底图尺寸不一致')
                self._overlay_frame = frame
            except KeyError:
                self._overlay_error = f'叠加层缺少帧 {self.number}，仅显示底图'
            except Exception as error:
                self._overlay_error = f'叠加层帧 {self.number} 读取失败：{error}'

    def overlay_settings_changed(self, *_):
        self._overlay_revision += 1
        self.render()

    def update_overlay_controls(self):
        present = self.overlay_dataset is not None
        self.overlay_source.setEnabled(self.index < 3 and self.dataset is not None
                                       and self.overlay_source.count() > 1)
        difference = self.dataset is not None and self.dataset.mask and len(self.dataset.shape) == 2
        self.overlay_mode.model().item(2).setEnabled(bool(difference))
        if not difference and self.overlay_mode.currentData() == 'difference':
            with QSignalBlocker(self.overlay_mode):
                self.overlay_mode.setCurrentIndex(0)
        for widget in (self.overlay_visible, self.overlay_mode, self.overlay_opacity, self.overlay_remove):
            widget.setEnabled(present)
        self.overlay_color.setEnabled(present and self.overlay_mode.currentData() != 'difference')
        self.overlay_percent.setText(f'{self.overlay_opacity.value()}%')
        self.overlay_toggle.setText('叠加 ●' if present else '叠加')
        if not present:
            text = '按真实帧号叠加；底图和掩膜需具有相同尺寸与方向'
        elif not self.overlay_visible.isChecked():
            text = '叠加层已隐藏；勾选「显示」恢复'
        elif self._frame is None:
            text = f'底图帧 {self.number} 不可用，不单独显示叠加层'
        elif self._overlay_error:
            text = self._overlay_error
        elif self.overlay_mode.currentData() == 'difference':
            a, b = foreground(self._frame), foreground(self._overlay_frame)
            text = (f'帧 {self.number} · 浅灰共有 {np.count_nonzero(a & b)} · '
                    f'橙色仅底图 {np.count_nonzero(a & ~b)} · 紫色仅叠加 {np.count_nonzero(~a & b)}')
        else:
            text = f'帧 {self.number} · 背景透明 · {self.overlay_color.currentText()}为叠加掩膜'
        self.overlay_status.setText(text)
        self.overlay_status.setToolTip(text + '\n尺寸相同不代表已经配准；请确保同一视野、方向与真实帧号。')

    def activate_plane(self, axis):
        self._active_plane = axis
        self.chosen.emit(self.index)
        self.render()

    def change_view_mode(self, mode):
        self.primary_panel.setVisible(mode in (0, 1))
        self.secondary_splitter.setVisible(mode in (0, 2, 3))
        self.row_panel.setVisible(mode in (0, 2))
        self.column_panel.setVisible(mode in (0, 3))
        self._active_plane = max(0, mode - 1)
        if self.dataset is not None:
            if mode == 1 and self._orth_requested is not None:
                self.request_sections()
            if mode != 1 and self._orth_data is None and self._orth_requested is None:
                self.request_sections()
            self.render()

    def set_orthogonal_position(self, normalized, notify=True):
        if self.dataset is None:
            return
        self._has_pixel_pick = True
        height, width = self.dataset.shape[:2]
        column = min(width - 1, max(0, int(normalized[0] * width)))
        row = min(height - 1, max(0, int(normalized[1] * height)))
        changed = (row, column) != (self.row_spin.value(), self.column_spin.value())
        with QSignalBlocker(self.row_spin), QSignalBlocker(self.column_spin):
            self.row_spin.setValue(row)
            self.column_spin.setValue(column)
        if changed:
            self.coordinate_changed(notify=notify)
        else:
            self.update_navigation()

    def pick_primary_position(self, point):
        self.set_orthogonal_position(point)

    def pick_section_position(self, axis, point):
        if self.dataset is None or self._orth_data is None:
            return
        indices = self._orth_data[2]
        number = indices[min(len(indices) - 1, max(0, int(point[1] * len(indices))))]
        height, width = self.dataset.shape[:2]
        col, row = self.column_spin.value(), self.row_spin.value()
        if axis == 1:
            col = min(width - 1, max(0, int(point[0] * width)))
        else:
            row = min(height - 1, max(0, int(point[0] * height)))
        self.set_orthogonal_position(((col + .5) / width, (row + .5) / height))
        self.request_layer(number)

    def step_section(self, axis, step):
        spin = self.row_spin if axis == 1 else self.column_spin
        spin.setValue(spin.value() + step)

    def coordinate_changed(self, *_, notify=True):
        if self.dataset is None:
            return
        self._has_pixel_pick = True
        self.update_navigation()
        self.request_sections()
        if notify:
            height, width = self.dataset.shape[:2]
            self.orthogonalChanged.emit(self.index, ((self.column_spin.value() + .5) / width,
                                                    (self.row_spin.value() + .5) / height))

    def request_sections(self):
        self._orth_token += 1
        if self.orth_job is not None:
            self.orth_job.cancelled.set()
        if self.dataset is None:
            return
        # Remove previous overlay pixels while a replacement is being read.
        if self._overlay_sections is not None:
            self._overlay_sections = None
            self._orth_signature = None
            self.render_sections()
        if self.view_mode.currentIndex() == 1:
            self._orth_requested = None
            self._orth_data = None
            self._orth_signature = None
            self._orth_timer.stop()
            self.section_status.setText(f'N={self.number} · Y={self.row_spin.value()} X={self.column_spin.value()} · 原始切片')
            return
        self._orth_requested = (self._orth_token, SequenceSpec.from_dataset(self.dataset, cache_sections=True),
                                self.row_spin.value(), self.column_spin.value(),
                                SequenceSpec.from_dataset(self.overlay_dataset, cache_sections=True)
                                if self.overlay_dataset is not None else None)
        self._orth_data = None
        self._orth_signature = None
        for view in (self.row_view, self.column_view):
            view.hint = '正在读取截面…'
            if not view.has_image():
                view.show_pixels(None)
        self.section_status.setText(f'读取截面… Y={self.row_spin.value()} X={self.column_spin.value()}')
        self._orth_timer.start()

    def start_sections(self):
        if self.orth_job is not None or self._orth_requested is None:
            return
        token, spec, row, column, overlay_spec = self._orth_requested
        job = SectionJob(token, spec, row, column, overlay_spec)
        self.orth_job = job
        job.signals.finished.connect(lambda t, result, error, j=job: self.sections_finished(j, t, result, error))
        job.signals.progress.connect(self.section_progress)
        QThreadPool.globalInstance().start(job)

    def section_progress(self, token, done, total):
        if token == self._orth_token and self.dataset is not None:
            self.section_status.setText(f'读取截面 {done}/{total} · Y={self.row_spin.value()} X={self.column_spin.value()}')

    def sections_finished(self, job, token, result, error):
        if self.orth_job is job:
            self.orth_job = None
        if token == self._orth_token and self.dataset is not None:
            self._orth_requested = None
            if result is not None:
                self._orth_data = result
                self._overlay_sections = job.overlay_result
                rows, columns, indices, errors = result
                if job.overlay_error:
                    errors = [*errors, '叠加截面：' + job.overlay_error]
                elif job.overlay_result is not None:
                    errors = [*errors, *('叠加截面：' + e for e in job.overlay_result[3])]
                self.section_status.setText(f'N={indices[0]}…{indices[-1]} · Y={job.row} X={job.column}' +
                                            (f' · {len(errors)} 层异常' if errors else ' · 原始数组顺序'))
                self.section_status.setToolTip('\n'.join(errors[:8]) or
                    '数组 [N,Y,X]：N 按文件层号递增，缺层留黑；X 向右、Y 向下。未翻转或转置。')
                self.render()
                self.update_navigation()
            elif error:
                self.section_status.setText('截面读取失败')
                self.section_status.setToolTip(error)
                for view in (self.row_view, self.column_view):
                    view.hint = error
                    view.show_pixels(None)
        if self._orth_requested is not None and not self._orth_timer.isActive():
            self.start_sections()

    def update_navigation(self):
        if self.dataset is None:
            return
        height, width = self.dataset.shape[:2]
        row, col = self.row_spin.value(), self.column_spin.value()
        self.update_pixel_value()
        if self.view_mode.currentIndex() == 1:
            self.section_status.setText(f'N={self.number} · Y={row} X={col} · 原始切片')
        # The primary image already has click/geometric markers. A second
        # persistent navigation cross would reappear blue beneath old red picks.
        self.view.set_navigation_point()
        indices = self._orth_data[2] if self._orth_data is not None else section_indices(self.dataset.indices)
        position = np.searchsorted(indices, self.number)
        if position >= len(indices) or indices[position] != self.number:
            self.row_view.set_navigation_point()
            self.column_view.set_navigation_point()
            return
        n = (min(len(indices) - 1, position) + .5) / len(indices)
        self.row_view.set_navigation_point(((col + .5) / width, n))
        self.column_view.set_navigation_point(((row + .5) / height, n))

    def update_pixel_value(self):
        if self.dataset is None or not self._has_pixel_pick:
            self.pixel_value.setText('值：—')
            self.pixel_value.setToolTip('点击图像显示原始像素值（调窗和掩膜显示之前）')
            return
        row, col = self.row_spin.value(), self.column_spin.value()
        coordinates = f'[N,Y,X]=[{self.number},{row},{col}]'
        if self._frame is None:
            missing = self.number not in self.dataset.indices
            self.pixel_value.setText('值：缺层' if missing else '值：不可用')
            self.pixel_value.setToolTip(coordinates + '\n当前图像未能读取，没有有效像素值')
            return
        value = self._frame[row, col]
        if np.ndim(value):
            raw = tuple(np.asarray(value).tolist())
            text = str(raw)
        else:
            raw = value.item()
            text = format(raw, '.10g') if isinstance(raw, float) else str(raw)
        self.pixel_value.setText(f'值：{text}')
        self.pixel_value.setToolTip(f'{coordinates}\ndtype={self._frame.dtype}\n'
                                   f'原始值={raw!r}\n调窗和掩膜显示之前的像素值')
        if self.overlay_dataset is not None:
            overlay_value = (self._overlay_frame[row, col].item() if self._overlay_frame is not None else None)
            self.pixel_value.setText(f'值：{text} · 掩膜：{overlay_value if overlay_value is not None else "缺帧/异常"}')
            self.pixel_value.setToolTip(self.pixel_value.toolTip() +
                f'\n叠加掩膜 {self.overlay_dataset.name}：{overlay_value!r}')

    def active_frame(self):
        if self._active_plane == 0:
            return self._frame
        return self._orth_data[self._active_plane - 1] if self._orth_data is not None else None

    def render_sections(self):
        if self.dataset is None or self._orth_data is None:
            return
        signature = (id(self._orth_data), self.ww.value(), self.wl.value(), self.dataset.mask,
                     self.threshold_preview, self._overlay_revision, id(self._overlay_sections))
        if signature == self._orth_signature:
            return
        self._orth_signature = signature
        for axis, (view, plane) in enumerate(zip((self.row_view, self.column_view), self._orth_data[:2])):
            overlay = self._overlay_sections[axis] if self._overlay_sections is not None else None
            view.show_pixels(self.display_frame(plane, overlay))

    def preview_available(self):
        return (self.index == 3 and self.dataset is not None
                and self.dataset.domain == 'reconstruction' and not self.dataset.mask
                and len(self.dataset.shape) == 2)

    def preview_active(self):
        return self.threshold_preview is not None and self.preview_available()

    def set_threshold_preview(self, threshold=None, comparison='ge'):
        settings = (threshold, comparison) if threshold is not None and self.preview_available() else None
        if settings == self.threshold_preview:
            return
        self.threshold_preview = settings
        if self.dataset is not None:
            self.render()

    def display_frame(self, frame, overlay=None):
        if frame is None:
            return None
        if self.preview_active():
            return binary_threshold(frame, *self.threshold_preview).astype(np.uint8) * 255
        pixels = display_pixels(frame, self.ww.value(), self.wl.value(), self.dataset.mask)
        if overlay is not None and self.overlay_dataset is not None and self.overlay_visible.isChecked():
            pixels = composite_mask(pixels, overlay, self.overlay_opacity.value()/100,
                self.overlay_color.currentData(), mode=self.overlay_mode.currentData(), base=frame)
        return pixels

    def set_active(self, active):
        self.setProperty('active', active)
        self.style().unpolish(self)
        self.style().polish(self)

    def clear(self):
        self._has_pixel_pick = False
        self.threshold_preview = None
        self._orth_token += 1
        self._orth_timer.stop()
        self._orth_requested = None
        self._orth_data = None
        self._orth_signature = None
        if self.orth_job is not None:
            self.orth_job.cancelled.set()
        self.dataset = None
        self._frame = None
        self.clear_overlay(render=False)
        self.overlay_toggle.setChecked(False)
        self.update_overlay_controls()
        self.title.setText('尚未放置数据')
        self.title.setToolTip('')
        self.primary_panel.label.setText('A · X→ / Y↓')
        self.row_panel.label.setText('B · X→ / N↓')
        self.column_panel.label.setText('C · Y→ / N↓')
        self.view.hint = ('拖入重建图像或体数据\n高级重建功能预留' if self.index == 3 else
                          '拖入文件、图像序列文件夹\n或左侧列表中的数据')
        self.view.show_pixels(None, reset=True)
        self.view.set_markers()
        self.view.set_reference_point()
        self.view.set_ray_trace()
        for view in self.section_views:
            view.set_navigation_point()
        for view in (self.row_view, self.column_view):
            view.hint = '拖入序列或体数据以显示截面'
            view.show_pixels(None, reset=True)
            view.set_markers()
        for spin in (self.row_spin, self.column_spin):
            with QSignalBlocker(spin):
                spin.setRange(0, 0)
            spin.setEnabled(False)
        self.section_status.setText('数组轴：N（层/视角）、Y（行）、X（列）')
        self.update_pixel_value()
        for widget in (self.layer, self.slider):
            with QSignalBlocker(widget):
                widget.setRange(0, 0)
            widget.setEnabled(False)
        for widget in (self.ww, self.wl, self.auto_button):
            widget.setEnabled(False)
        self.count.setText('—')
        self.info.setText('左键定位/框选 · 中键平移 · 右键调窗')
        self.bindingChanged.emit()

    def assign(self, dataset, number=None):
        self.clear_overlay(render=False)
        self._active_plane = 0
        self._has_pixel_pick = False
        self.dataset = dataset
        self.title.setText(dataset.name)
        self.title.setToolTip(str(dataset.paths[0]))
        height, width = dataset.shape[:2]
        for spin, count in ((self.row_spin, height), (self.column_spin, width)):
            with QSignalBlocker(spin):
                spin.setRange(0, count - 1)
                spin.setValue(count // 2)
            spin.setEnabled(count > 1)
        axis = '层' if dataset.domain == 'reconstruction' else '视角'
        self.primary_panel.label.setText('A · 原始切片 X→ / Y↓' if dataset.domain == 'reconstruction' else 'A · 投影帧 X→ / Y↓')
        self.row_panel.label.setText(f'B · X→ / N({axis})↓')
        self.column_panel.label.setText(f'C · Y→ / N({axis})↓')
        self._orth_data = None
        self._orth_signature = None
        for view in (self.row_view, self.column_view):
            view.show_pixels(None, reset=True)
        for widget in (self.layer, self.slider):
            with QSignalBlocker(widget):
                widget.setRange(min(dataset.indices), max(dataset.indices))
            widget.setEnabled(len(dataset.indices) > 1)
        self.count.setText(f'{len(dataset.indices)} 层')
        self.set_layer(dataset.indices[0] if number is None else number, reset=True)
        self.reset_window()
        self.request_sections()
        self.bindingChanged.emit()

    def request_layer(self, number):
        self.chosen.emit(self.index)
        self.layerChanged.emit(self.index, number)

    def step(self, amount):
        if self.dataset is None:
            return
        # Wheel moves one numeric layer; absent IDs are intentionally visible.
        self.request_layer(max(min(self.dataset.indices),
                               min(max(self.dataset.indices), self.number + amount)))

    def set_layer(self, number, reset=False):
        if self.dataset is None:
            return
        self.number = number
        for widget in (self.layer, self.slider):
            with QSignalBlocker(widget):
                widget.setValue(number)
        self._frame = None
        try:
            self._frame = self.dataset.frame(number)
            shape = ' × '.join(str(n) for n in self._frame.shape[:2])
            self.info.setText(f'层号 {number}   ·   {shape}   ·   {self._frame.dtype}' +
                              ('   ·   二值掩膜' if self.dataset.mask else ''))
        except KeyError:
            self.view.hint = f'层号 {number} 缺失\n此数据没有对应层，未使用邻层代替'
            self.info.setText(f'层号 {number} 缺失   ·   本序列 {min(self.dataset.indices)}–{max(self.dataset.indices)}')
        except Exception as error:
            self.view.hint = f'层号 {number} 读取失败\n{error}'
            self.info.setText(f'层号 {number} 读取失败')
        self.load_overlay_frame()
        self.render(reset=reset)
        self.update_navigation()

    def reset_window(self):
        frame = self.active_frame()
        if frame is None:
            return
        width, level = auto_window(frame, self.section_views[self._active_plane].selection)
        with QSignalBlocker(self.ww), QSignalBlocker(self.wl):
            self.ww.setValue(width)
            self.wl.setValue(level)
            self.ww.setSingleStep(max(.000001, width / 100))
            self.wl.setSingleStep(max(.000001, width / 100))
        self.render()

    def drag_window(self, dx, dy):
        if not self.ww.isEnabled():
            return
        width = self.ww.value()
        with QSignalBlocker(self.ww), QSignalBlocker(self.wl):
            self.ww.setValue(width * np.exp(np.clip(dx * .008, -2, 2)))
            self.wl.setValue(self.wl.value() - dy * width / 200)
        self.render()

    def render(self, *_, reset=False):
        if self.dataset is None:
            return
        self.update_overlay_controls()
        active = self.active_frame()
        preview = self.preview_active()
        gray = active is not None and active.ndim == 2 and not self.dataset.mask and not preview
        for widget in (self.ww, self.wl, self.auto_button):
            widget.setEnabled(gray)
        axis = '层' if self.dataset.domain == 'reconstruction' else '视角'
        if preview:
            threshold, comparison = self.threshold_preview
            rule = f'{"≥" if comparison == "ge" else "≤"}{threshold:g}'
            self.primary_panel.label.setText(f'A · 阈值预览 {rule} · X→ / Y↓')
        else:
            self.primary_panel.label.setText('A · 原始切片 X→ / Y↓' if axis == '层' else 'A · 投影帧 X→ / Y↓')
        mode = '阈值预览 · ' if preview else ''
        self.row_panel.label.setText(f'B · {mode}X→ / N({axis})↓')
        self.column_panel.label.setText(f'C · {mode}Y→ / N({axis})↓')
        self.view.show_pixels(self.display_frame(self._frame, self._overlay_frame), reset)
        self.render_sections()


class DatasetDelegate(QStyledItemDelegate):
    def sizeHint(self, option, index):
        lines = str(index.data() or '').splitlines()
        return QSize(200, max(54, len(lines) * option.fontMetrics.height() + 20))

    def paint(self, painter, option, index):
        displayed = bool(index.data(DISPLAY_PANES_ROLE))
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.save()
        rect = option.rect.adjusted(1, 1, -1, -1)
        painter.setBrush(QColor('#254b49' if displayed else '#26384d' if selected else '#1c2a3b'))
        painter.setPen(QPen(QColor('#65d9c9' if displayed else '#7b92ae' if selected else '#1c2a3b'), 1))
        painter.drawRoundedRect(rect, 4, 4)
        painter.setFont(option.font)
        painter.setPen(QColor('#b4f1e7' if displayed else '#dbe5f0'))
        fm = option.fontMetrics
        for number, line in enumerate(str(index.data() or '').splitlines()):
            text = fm.elidedText(line, Qt.TextElideMode.ElideRight, max(1, rect.width() - 20))
            painter.drawText(rect.left() + 10, rect.top() + 10 + fm.ascent() + number * fm.height(), text)
        if selected:
            painter.setPen(QPen(QColor('#e3ecef'), 2))
            painter.drawLine(rect.topLeft() + QPoint(3, 6), rect.bottomLeft() + QPoint(3, -6))
        painter.restore()


class DatasetList(QListWidget):
    filesDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setSpacing(5)
        self.setMinimumHeight(110)
        self.setWordWrap(True)
        self.setItemDelegate(DatasetDelegate(self))

    def startDrag(self, actions):
        item = self.currentItem()
        if item is None:
            return
        mime = QMimeData()
        mime.setData(MIME_DATASET, item.data(Qt.ItemDataRole.UserRole).encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)

    def dragEnterEvent(self, event):
        if local_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        self.dragEnterEvent(event)

    def dropEvent(self, event):
        paths = local_paths(event.mimeData())
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()


class LoadSignals(QObject):
    finished = Signal(int, int, object, object)


class LoadJob(QRunnable):
    def __init__(self, epoch, target, paths, options):
        super().__init__()
        self.epoch, self.target = epoch, target
        self.paths, self.options = paths, options
        self.cancelled = Event()
        self.signals = LoadSignals()

    def run(self):
        datasets, errors = load_datasets(self.paths, self.options, self.cancelled)
        self.signals.finished.emit(self.epoch, self.target, datasets, errors)


class ProcessingSignals(QObject):
    progress = Signal(str)
    finished = Signal(object, object)


class PercentileWindowJob(QRunnable):
    def __init__(self, epoch, dataset, pane_index):
        super().__init__()
        self.epoch, self.source_id, self.pane_index = epoch, dataset.id, pane_index
        self.source = ProcessingSource.from_dataset(dataset)
        self.cancelled = Event()
        self.signals = ProcessingSignals()

    def run(self):
        result, error = None, None
        try:
            result = collect_statistics(self.source.sequence, self.cancelled, self.signals.progress.emit)
        except ProcessingCancelled:
            pass
        except Exception as exception:
            error = str(exception)
        finally:
            self.source = None
        self.signals.finished.emit(result, error)


def collapse_button(text, content):
    toggle = QToolButton()
    toggle.setObjectName('toolSectionToggle')
    toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    toggle.setMinimumHeight(44)
    toggle.setText(text)
    toggle.setCheckable(True)
    toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    toggle.setArrowType(Qt.ArrowType.RightArrow)
    toggle.toggled.connect(content.setVisible)
    toggle.toggled.connect(lambda checked: toggle.setArrowType(
        Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow))
    content.hide()
    return toggle


STYLE = '''
QMainWindow, QWidget { background: #111925; color: #dbe5f0; font-family: "Microsoft YaHei UI", "Segoe UI"; font-size: 12px; }
QFrame#pane { background: #141f2d; border: 1px solid #2a3a50; border-radius: 7px; }
QFrame#pane[active="true"] { border: 1px solid #55c9c0; }
QLabel#paneTag { color: #63d3c7; font-weight: 600; }
QLabel#muted { color: #8192a9; font-size: 11px; }
QLabel#heading { font-size: 21px; font-weight: 600; }
QLabel#sidebarTitle { font-size: 16px; font-weight: 600; }
QPushButton { background: #233347; border: 1px solid #34475e; border-radius: 4px; padding: 5px 9px; }
QPushButton:hover { background: #31475e; border-color: #55c9c0; }
QPushButton:disabled { color: #5f7085; }
QPushButton#danger { color: #ffb8ae; background: #392a31; border-color: #6c4248; }
QPushButton#pageTab { padding: 8px 17px; font-size: 15px; font-weight: 600; background: #182636; }
QPushButton#pageTab:checked { background: #244c4c; border-color: #63d3c7; color: #dcfff5; }
QToolButton#toolSectionToggle { background: #253c50; color: #dcf8f3; border: 1px solid #47677d; border-left: 4px solid #63d3c7; border-radius: 5px; padding: 9px 8px; font-size: 14px; font-weight: 600; text-align: left; }
QToolButton#toolSectionToggle:hover { background: #305267; border-color: #7bddcf; }
QToolButton#toolSectionToggle:checked { background: #244c4c; color: #edfffa; border-color: #63d3c7; }
QToolButton#toolSectionToggle:focus { border-color: #b1f4e8; }
QListWidget { background: #0e1621; border: 1px dashed #3a506b; border-radius: 5px; padding: 5px; }
QListWidget::item { background: #1c2a3b; padding: 9px 5px; border-radius: 3px; }
QListWidget::item:selected { background: #254746; color: #a1f0e3; }
QSpinBox, QDoubleSpinBox, QComboBox { background: #0e1723; border: 1px solid #34465c; border-radius: 3px; padding: 3px; min-height: 19px; }
QSpinBox:disabled, QDoubleSpinBox:disabled { color: #5e6d80; }
QGroupBox { border: 1px solid #2b3e54; border-radius: 5px; margin-top: 12px; padding: 9px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #91a6be; }
QSlider::groove:horizontal { height: 4px; background: #2b4058; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #4dbbb4; }
QSlider::handle:horizontal { background: #83e1d4; width: 12px; margin: -5px 0; border-radius: 6px; }
QSplitter::handle { background: #111925; width: 7px; }
QStatusBar { color: #91a6be; }
'''


class ReviewWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.datasets = {}
        self.active_pane = 0
        self.link_number = 0
        self.epoch = 0
        self.jobs = []
        self.percentile_job = None
        self._percentile_target_key = None
        self._percentile_cached_id = None
        self._percentile_statistics = None
        self._percentile_frame_key = None
        self._percentile_frame_statistics = None
        self._percentile_timer = QTimer(self)
        self._percentile_timer.setSingleShot(True)
        self._percentile_timer.setInterval(180)
        self._percentile_timer.timeout.connect(self.apply_percentile_window)
        self._updating_details = False
        self.reference = None
        self.ray_reference = None
        self._geometry_cache = {}
        self.maximized_pane = None
        self._initial_window_placed = False
        self._startup_target_screen = None
        self._startup_placement_timer = QTimer(self)
        self._startup_placement_timer.setSingleShot(True)
        self._startup_placement_timer.timeout.connect(self.finish_initial_placement)
        self.setWindowTitle('数据审查 · Projection Review')
        self.resize(1680, 960)
        self.setMinimumSize(1360, 780)
        self.setAcceptDrops(True)
        self.setStyleSheet(STYLE)
        shell = QWidget()
        self.setCentralWidget(shell)
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(16, 13, 16, 9)
        root = QWidget()
        self.review_page = root
        self.pages = QStackedWidget()
        self.pages.addWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        header = QHBoxLayout()
        self.review_tab = WorkspaceTab('数据审查')
        self.reconstruction_tab = WorkspaceTab('投影域重建')
        self.labeling_tab = WorkspaceTab('标签勾画')
        self.review_tab.clicked.connect(lambda: self.select_page(0))
        self.reconstruction_tab.clicked.connect(lambda: self.select_page(1))
        self.labeling_tab.clicked.connect(lambda: self.select_page(2))
        self.review_tab.dragHovered.connect(lambda: self.select_page(0))
        self.reconstruction_tab.setAcceptDrops(False)
        self.labeling_tab.dragHovered.connect(lambda: self.select_page(2))
        self.review_tab.dropped.connect(lambda mime: self.receive_drop(self.active_pane, mime))
        self.labeling_tab.dropped.connect(lambda mime: self.labeling_page.receive_drop(mime))
        for tab in (self.review_tab, self.reconstruction_tab, self.labeling_tab):
            tab.setObjectName('pageTab')
            tab.setCheckable(True)
            header.addWidget(tab)
        self.review_tab.setChecked(True)
        subtitle = QLabel('单样本工作台  /  拖入 · 对齐 · 比较')
        subtitle.setObjectName('muted')
        header.addWidget(subtitle)
        header.addStretch()
        self.link = QCheckBox('投影联动绑定')
        self.link.setToolTip('绑定切层、定位、矩形、缩放和平移；窗宽窗位各自独立')
        self.link.setChecked(True)
        self.link.toggled.connect(self.toggle_link)
        clear = button('一键清空', self.request_clear_all)
        clear.setObjectName('danger')
        clear.setToolTip('移除当前样本的全部数据和待完成导入，不修改磁盘文件')
        shell_layout.addLayout(header)
        # On a small/high-DPI screen the workspace may need more logical pixels
        # than the desktop offers. Scroll the content instead of forcing the
        # top-level window beyond the screen's work area.
        self.workspace_scroll = QScrollArea()
        self.workspace_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.workspace_scroll.setWidgetResizable(True)
        self.workspace_scroll.setWidget(self.pages)
        shell_layout.addWidget(self.workspace_scroll, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.workspace_splitter = splitter
        splitter.setChildrenCollapsible(False)
        board = QWidget()
        grid = QGridLayout(board)
        self.board_grid = grid
        grid.setContentsMargins(0, 7, 0, 0)
        grid.setSpacing(10)
        self.panes = [ReviewPane(i) for i in range(4)]
        for i, pane in enumerate(self.panes):
            grid.addWidget(pane, i // 2, i % 2)
            pane.chosen.connect(self.select_pane)
            pane.maximizeRequested.connect(self.toggle_pane_maximize)
            pane.layerChanged.connect(self.change_layer)
            pane.dropReceived.connect(self.receive_drop)
            pane.overlaySourceChanged.connect(self.overlay_source_changed)
            pane.overlayToolsRequested.connect(self.show_overlay_tools)
            pane.bindingChanged.connect(self.update_link_range)
            pane.bindingChanged.connect(self.update_displayed_items)
            pane.bindingChanged.connect(self.update_percentile_actions)
            pane.bindingChanged.connect(self.refresh_reference_points)
            pane.bind.toggled.connect(lambda checked, p=pane: self.align_binding(p, checked))
            pane.view.markersChanged.connect(lambda crosshair, selection, idx=i:
                                              self.sync_markers(idx, crosshair, selection))
            pane.view.viewChanged.connect(lambda state, idx=i: self.sync_view(idx, state))
            pane.orthogonalChanged.connect(self.sync_orthogonal_position)
            pane.auxiliaryViewChanged.connect(self.sync_auxiliary_view)
            for view in pane.section_views:
                view.selectionStarted.connect(self.clear_position_overlays)
            if i == 3:
                pane.view.pointPicked.connect(self.pick_reconstruction_point)
            else:
                pane.view.pointPicked.connect(lambda point, idx=i: self.pick_projection_ray(idx, point))
        for i in range(2):
            grid.setRowStretch(i, 1)
            grid.setColumnStretch(i, 1)
        sidebar = QWidget()
        self.image_sidebar = sidebar
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(360)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 7, 10, 0)
        title = QLabel('图像管理')
        title.setObjectName('sidebarTitle')
        side.addWidget(title)
        hint = QLabel('拖入文件 / 序列文件夹\n再将列表条目拖到对应画布')
        hint.setObjectName('muted')
        side.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(button('添加文件', self.open_files))
        row.addWidget(button('添加文件夹', self.open_folder))
        side.addLayout(row)
        self.dataset_list = DatasetList()
        self.dataset_list.filesDropped.connect(self.import_paths)
        self.dataset_list.currentItemChanged.connect(self.show_details)
        self.dataset_list.itemDoubleClicked.connect(self.place_selected)
        side.addWidget(self.dataset_list, 1)
        self.details = QLabel('暂无数据')
        self.details.setWordWrap(True)
        self.details.setObjectName('muted')
        side.addWidget(self.details)
        row = QHBoxLayout()
        self.domain = QComboBox()
        self.domain.addItem('投影域', 'projection')
        self.domain.addItem('重建域', 'reconstruction')
        self.domain.currentIndexChanged.connect(self.edit_domain)
        row.addWidget(self.domain)
        self.mask = QCheckBox('二值掩膜')
        self.mask.setToolTip('有限且非零的像素显示为白色；不改动源数据')
        self.mask.toggled.connect(self.edit_mask)
        row.addWidget(self.mask)
        side.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(button('放入选中画布', self.place_selected))
        row.addWidget(button('移除数据', self.remove_selected))
        side.addLayout(row)
        splitter.addWidget(sidebar)
        center = QWidget()
        center.setMinimumWidth(760)
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.addWidget(board, 1)
        splitter.addWidget(center)
        self.tools_sidebar = QScrollArea()
        self.tools_sidebar.setMinimumWidth(280)
        self.tools_sidebar.setMaximumWidth(390)
        self.tools_sidebar.setFrameShape(QFrame.Shape.NoFrame)
        self.tools_sidebar.setWidgetResizable(True)
        self.tools_sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        tool_content = QWidget()
        tools = QVBoxLayout(tool_content)
        tools.setContentsMargins(10, 7, 0, 0)
        tools.setSpacing(10)
        tools_title = QLabel('工具')
        tools_title.setObjectName('sidebarTitle')
        tools.addWidget(tools_title)
        self.view_tools_panel = QWidget()
        view_tools = QVBoxLayout(self.view_tools_panel)
        view_tools.setContentsMargins(4, 4, 4, 8)
        view_tools.setSpacing(8)
        view_tools.addWidget(self.link)
        row = QHBoxLayout()
        row.addWidget(button('清除标记', self.clear_markers))
        row.addWidget(button('全部适应', self.fit_all))
        view_tools.addLayout(row)
        view_tools.addWidget(clear)
        self.geometry_enabled = QCheckBox('几何定位：重建点 / 投影射线')
        self.geometry_enabled.setChecked(True)
        self.geometry_enabled.setToolTip('点击重建图像：投影对应点；点击投影图像：重建空间射线和当前切片交点')
        self.geometry_enabled.toggled.connect(self.refresh_reference_points)
        view_tools.addWidget(self.geometry_enabled)
        self.geometry_info = QLabel('点击重建：投影对应点\n点击投影：重建射线 / 切片交点')
        self.geometry_info.setWordWrap(True)
        self.geometry_info.setObjectName('muted')
        view_tools.addWidget(self.geometry_info)
        self.view_tools_toggle = collapse_button('查看与定位', self.view_tools_panel)
        tools.addWidget(self.view_tools_toggle)
        tools.addWidget(self.view_tools_panel)
        self.percentile_panel = QWidget()
        percentile_form = QFormLayout(self.percentile_panel)
        percentile_form.setContentsMargins(4, 4, 4, 8)
        percentile_form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.percentile_target_label = QLabel('画布 01 · 未放置灰度图像')
        self.percentile_target_label.setWordWrap(True)
        self.percentile_target_label.setMinimumWidth(0)
        percentile_form.addRow(self.percentile_target_label)
        self.percentile_lower, self.percentile_upper = QDoubleSpinBox(), QDoubleSpinBox()
        for widget, value in ((self.percentile_lower, 1.), (self.percentile_upper, 99.)):
            widget.setRange(0., 100.)
            widget.setDecimals(3)
            widget.setSingleStep(.1)
            widget.setSuffix(' %')
            widget.setMinimumWidth(0)
            widget.setKeyboardTracking(False)
            widget.setValue(value)
            widget.valueChanged.connect(self.schedule_percentile_window)
        percentile_form.addRow('下百分位', self.percentile_lower)
        percentile_form.addRow('上百分位', self.percentile_upper)
        self.percentile_scope = QComboBox()
        self.percentile_scope.addItem('当前视图', 'frame')
        self.percentile_scope.addItem('整组数据（全局）', 'sequence')
        self.percentile_scope.currentIndexChanged.connect(self.schedule_percentile_window)
        percentile_form.addRow('统计范围', self.percentile_scope)
        self.percentile_live = QCheckBox('调整百分位时实时预览')
        self.percentile_live.setChecked(True)
        self.percentile_live.toggled.connect(self.schedule_percentile_window)
        percentile_form.addRow(self.percentile_live)
        self.percentile_apply = button('计算并调窗（仅当前幕布）', self.apply_percentile_window)
        percentile_form.addRow(self.percentile_apply)
        self.percentile_cancel = button('取消整组统计', self.cancel_percentile_window)
        self.percentile_cancel.hide()
        percentile_form.addRow(self.percentile_cancel)
        self.percentile_result = QLabel('点击幕布选择目标，再调整百分位或点击计算。')
        self.percentile_result.setWordWrap(True)
        self.percentile_result.setMinimumWidth(0)
        self.percentile_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        percentile_form.addRow(self.percentile_result)
        percentile_hint = QLabel('按原始灰度统计，不受框选、调窗和叠加影响。\n窗宽=上界−下界；窗位=(上界+下界)/2。\n切层后保留结果，不随切层自动重算。')
        percentile_hint.setWordWrap(True)
        percentile_hint.setObjectName('muted')
        percentile_form.addRow(percentile_hint)
        self.percentile_toggle = collapse_button('百分位调窗（测试）', self.percentile_panel)
        tools.addWidget(self.percentile_toggle)
        tools.addWidget(self.percentile_panel)
        self.overlay_tools_panel = QWidget()
        overlay_tools = QVBoxLayout(self.overlay_tools_panel)
        overlay_tools.setContentsMargins(4, 4, 4, 8)
        self.overlay_target = QLabel('画布 01 · 投影叠加')
        overlay_tools.addWidget(self.overlay_target)
        self.overlay_stack = QStackedWidget()
        for pane in self.panes[:3]:
            self.overlay_stack.addWidget(pane.overlay_panel)
            pane.overlay_panel.show()
        placeholder = QLabel('选择前三个投影幕布之一，或点击其「叠加」按钮。')
        placeholder.setWordWrap(True)
        self.overlay_stack.addWidget(placeholder)
        overlay_tools.addWidget(self.overlay_stack)
        self.overlay_tools_toggle = collapse_button('投影叠加（当前幕布）', self.overlay_tools_panel)
        self.overlay_tools_toggle.toggled.connect(self.sync_overlay_tools)
        tools.addWidget(self.overlay_tools_toggle)
        tools.addWidget(self.overlay_tools_panel)
        options = QGroupBox()
        form = QFormLayout(options)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.import_domain = QComboBox()
        self.import_domain.addItem('投影域', 'projection')
        self.import_domain.addItem('重建域', 'reconstruction')
        form.addRow('默认数据域', self.import_domain)
        self.import_mask = QCheckBox('作为二值掩膜读取')
        form.addRow(self.import_mask)
        dims = QHBoxLayout()
        self.raw_width, self.raw_height = QSpinBox(), QSpinBox()
        for spin in (self.raw_width, self.raw_height):
            spin.setRange(1, 32768)
            spin.setMinimumWidth(0)
            spin.setValue(768)
        dims.addWidget(self.raw_width)
        dims.addWidget(QLabel('×'))
        dims.addWidget(self.raw_height)
        form.addRow('投影 宽 × 高', dims)
        recon_dims = QHBoxLayout()
        self.recon_width, self.recon_height = QSpinBox(), QSpinBox()
        for spin in (self.recon_width, self.recon_height):
            spin.setRange(1, 32768)
            spin.setMinimumWidth(0)
            spin.setValue(800)
            recon_dims.addWidget(spin)
            if spin is self.recon_width:
                recon_dims.addWidget(QLabel('×'))
        form.addRow('重建 宽 × 高', recon_dims)
        self.auto_recon_shape = QCheckBox('自动识别重建尺寸')
        self.auto_recon_shape.setToolTip('先读 ImageParam.ini / XML；无元数据的方形 SLC 按字节数和所选 dtype 推导')
        self.auto_recon_shape.setChecked(True)
        self.auto_recon_shape.toggled.connect(lambda checked:
            [spin.setEnabled(not checked) for spin in (self.recon_width, self.recon_height)])
        for spin in (self.recon_width, self.recon_height):
            spin.setEnabled(False)
        form.addRow(self.auto_recon_shape)
        self.raw_dtype = QComboBox()
        for label, dtype in [('uint16 · 小端', '<u2'), ('float32 · 小端', '<f4'),
                             ('float16 · 小端', '<f2'), ('uint8', '|u1'),
                             ('int16 · 小端', '<i2'), ('uint16 · 大端', '>u2'),
                             ('float32 · 大端', '>f4')]:
            self.raw_dtype.addItem(label, dtype)
        self.raw_dtype.setToolTip('投影和其他 Raw 使用此选项；slice 文件夹及 SLC 原始数据固定按小端 int16 读取')
        form.addRow('其他 Raw dtype', self.raw_dtype)
        self.use_xml = QCheckBox('优先读取附近 XML 的尺寸')
        self.use_xml.setChecked(True)
        form.addRow(self.use_xml)
        raw_hint = QLabel('slice / SLC：16bit signed（小端 int16）\nNPY：[N,H,W]；图像文件夹按末尾数字排序\n无层号的序列与体数据从 0 编号')
        raw_hint.setWordWrap(True)
        raw_hint.setObjectName('muted')
        form.addRow(raw_hint)
        self.import_options_panel = options
        self.import_options_toggle = collapse_button('新数据导入参数', options)
        tools.addWidget(self.import_options_toggle)
        tools.addWidget(options)
        help_label = QLabel('绑定时：切层 / 定位 / 框选 / 缩放 / 平移\n调窗独立；有框时按框内像素自动调窗\nslice / SLC 自动归入重建域')
        help_label.setObjectName('muted')
        help_label.setWordWrap(True)
        form.addRow(help_label)
        tools.addStretch()
        self.tools_sidebar.setWidget(tool_content)
        splitter.addWidget(self.tools_sidebar)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([250, 1080, 300])
        QTimer.singleShot(0, lambda: splitter.setSizes([250, 1080, 300]))
        layout.addWidget(splitter, 1)
        footer = QHBoxLayout()
        footer.addWidget(QLabel('投影联动层'))
        self.link_slider = QSlider(Qt.Orientation.Horizontal)
        self.link_slider.valueChanged.connect(self.change_link_layer)
        footer.addWidget(self.link_slider, 1)
        self.link_label = QLabel('未加载投影')
        footer.addWidget(self.link_label)
        center_layout.addLayout(footer)
        self.show_details()
        self.update_percentile_actions()
        self.update_link_range()
        self.statusBar().showMessage('就绪 · 仅审查当前样本，换样本前可一键清空')
        QShortcut(QKeySequence('F'), self, activated=self.fit_all)
        QShortcut(QKeySequence('Ctrl+O'), self, activated=self.open_files)
        QShortcut(QKeySequence('Esc'), self, activated=self.restore_panes)
        from .analysis_widget import ProjectionReconstructionWidget
        self.reconstruction_page = ProjectionReconstructionWidget(self)
        self.reconstruction_page.datasetGenerated.connect(self.add_generated_dataset)
        self.pages.addWidget(self.reconstruction_page)
        from .labeling_widget import LabelingWidget
        self.labeling_page = LabelingWidget(self)
        self.labeling_page.datasetGenerated.connect(self.add_generated_dataset)
        self.pages.addWidget(self.labeling_page)

    def startup_screen(self):
        app = QApplication.instance()
        return app.screenAt(QCursor.pos()) or self.screen() or app.primaryScreen()

    def place_initial_window(self, screen):
        geometry, frame = self.geometry(), self.frameGeometry()
        margins = QMargins(max(0, geometry.left()-frame.left()),
                           max(0, geometry.top()-frame.top()),
                           max(0, frame.right()-geometry.right()),
                           max(0, frame.bottom()-geometry.bottom()))
        rect, minimum = startup_geometry(screen.availableGeometry(), margins)
        self.setMinimumSize(minimum)
        self.setGeometry(rect)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_window_placed:
            # Run only on the first show. Later hide/show or page switches must
            # not undo the user's window size or position.
            self._initial_window_placed = True
            screen = self.startup_screen()
            if screen is not None:
                self.setScreen(screen)
                self.place_initial_window(screen)
                # Windows can update frame margins during a mixed-DPI screen
                # change. Correct once after that first layout has settled.
                self._startup_target_screen = screen
                self._startup_placement_timer.start(0)

    def finish_initial_placement(self):
        screen = self._startup_target_screen
        self._startup_target_screen = None
        if self.isVisible() and screen in QApplication.instance().screens():
            self.place_initial_window(screen)

    def percentile_pane(self):
        pane = self.panes[self.active_pane]
        if (pane.dataset is None or pane.dataset.mask or len(pane.dataset.shape) != 2
                or pane.preview_active()):
            return None
        return pane

    def update_percentile_actions(self, *_):
        if not hasattr(self, 'percentile_apply'):
            return
        pane = self.panes[self.active_pane]
        key = (self.active_pane, pane.dataset.id if pane.dataset is not None else None,
               pane.dataset.mask if pane.dataset is not None else None, pane.preview_active(), pane._active_plane)
        if key != self._percentile_target_key:
            self.cancel_percentile_window(notify=False)
            self._percentile_target_key = key
            self._percentile_frame_key = self._percentile_frame_statistics = None
            self.percentile_result.setText('调整百分位或点击计算；只修改此幕布。' if self.percentile_pane() is not None
                else '请选择已放置灰度原图的幕布；二值掩膜、彩色图和阈值预览不能调窗。')
        name = pane.dataset.name if pane.dataset is not None else '尚未放置数据'
        self.percentile_target_label.setText(f'目标幕布 {self.active_pane+1:02d} · {name}')
        valid_range = self.percentile_lower.value() < self.percentile_upper.value()
        ready = self.percentile_pane() is not None
        self.percentile_apply.setEnabled(ready and valid_range)
        self.percentile_live.setEnabled(ready)
        self.percentile_cancel.setVisible(self.percentile_job is not None)

    def schedule_percentile_window(self, *_):
        self._percentile_timer.stop()
        self.update_percentile_actions()
        if self.percentile_lower.value() >= self.percentile_upper.value():
            self.percentile_result.setText('下百分位必须小于上百分位，范围为 0%–100%。')
            return
        if self.percentile_scope.currentData() != 'sequence':
            self.cancel_percentile_window(notify=False)
        if not self.percentile_live.isChecked():
            self.cancel_percentile_window(notify=False)
        if self.percentile_live.isChecked() and self.percentile_pane() is not None:
            self._percentile_timer.start()

    def apply_percentile_window(self, *_):
        self._percentile_timer.stop()
        pane = self.percentile_pane()
        if pane is None:
            self.update_percentile_actions()
            return
        if not 0 <= self.percentile_lower.value() < self.percentile_upper.value() <= 100:
            self.percentile_result.setText('下百分位必须小于上百分位，范围为 0%–100%。')
            return
        if self.percentile_scope.currentData() == 'sequence':
            if self._percentile_cached_id == pane.dataset.id:
                self.apply_percentile_statistics(self._percentile_statistics, pane)
                return
            if self.percentile_job is not None:
                if (self.percentile_job.source_id == pane.dataset.id and
                        self.percentile_job.pane_index == self.active_pane):
                    return  # Apply the latest percentile settings when statistics finish.
                self.cancel_percentile_window(notify=False)
            job = PercentileWindowJob(self.epoch, pane.dataset, self.active_pane)
            self.percentile_job = job
            self.jobs.append(job)
            job.signals.progress.connect(lambda text, j=job: self.percentile_progress(j, text))
            job.signals.finished.connect(lambda result, error, j=job: self.percentile_finished(j, result, error))
            self.percentile_result.setText('正在后台统计整组原始灰度…')
            self.update_percentile_actions()
            QThreadPool.globalInstance().start(job)
        else:
            self.cancel_percentile_window(notify=False)
            frame = pane.active_frame()
            if frame is None:
                self.percentile_result.setText('当前视图尚未读取完成或缺帧，请切换到有效图像后重试。')
                return
            key = (pane.dataset.id, pane._active_plane,
                   pane.number if pane._active_plane == 0 else
                   (pane.row_spin.value(), pane.column_spin.value(), id(pane._orth_data)))
            try:
                if key != self._percentile_frame_key:
                    stats = IntensityStatistics.from_frame(frame)
                    self._percentile_frame_key, self._percentile_frame_statistics = key, stats
                self.apply_percentile_statistics(self._percentile_frame_statistics, pane)
            except ValueError as error:
                self.percentile_result.setText(str(error))

    def apply_percentile_statistics(self, statistics, pane):
        low_p, high_p = self.percentile_lower.value(), self.percentile_upper.value()
        try:
            low, high, width, level = statistics.window(low_p, high_p)
        except ValueError as error:
            self.percentile_result.setText(str(error))
            return
        with QSignalBlocker(pane.ww), QSignalBlocker(pane.wl):
            pane.ww.setValue(width)
            pane.wl.setValue(level)
            pane.ww.setSingleStep(max(.000001, width/100))
            pane.wl.setSingleStep(max(.000001, width/100))
        pane.render()
        scope = '整组数据' if self.percentile_scope.currentData() == 'sequence' else '当前视图'
        method = f'近似采样 {len(statistics.values):,} 个值' if statistics.approximate else '精确统计'
        text = (f'幕布 {pane.index+1:02d} · {scope} · {method}\n'
                f'P{low_p:g} = {low:.10g}\nP{high_p:g} = {high:.10g}\n'
                f'窗宽 = {pane.ww.value():.10g}\n窗位 = {pane.wl.value():.10g}\n'
                f'{statistics.frames} 帧 · 有限像素 {statistics.finite_count:,}')
        if high <= low:
            text += '\n百分位值相同，使用窗宽 1 避免除零。'
        self.percentile_result.setText(text)

    def percentile_progress(self, job, text):
        if job is self.percentile_job and job.epoch == self.epoch:
            self.percentile_result.setText(text)

    def percentile_finished(self, job, result, error):
        if job in self.jobs:
            self.jobs.remove(job)
        job.signals.progress.disconnect()
        job.signals.finished.disconnect()
        if job is not self.percentile_job or job.epoch != self.epoch:
            return
        self.percentile_job = None
        pane = self.percentile_pane()
        if error:
            self.percentile_result.setText(f'整组统计失败：{error}')
        elif (result is not None and pane is not None and pane.dataset.id == job.source_id
              and self.active_pane == job.pane_index and self.percentile_scope.currentData() == 'sequence'):
            self._percentile_cached_id, self._percentile_statistics = job.source_id, result
            self.apply_percentile_statistics(result, pane)
        self.update_percentile_actions()

    def cancel_percentile_window(self, *_, notify=True):
        self._percentile_timer.stop()
        if self.percentile_job is not None:
            self.percentile_job.cancelled.set()
            self.percentile_job = None
            if notify:
                self.percentile_result.setText('整组统计已取消，保留原来的窗宽窗位。')
        if hasattr(self, 'percentile_cancel'):
            self.percentile_cancel.hide()

    def select_page(self, index):
        if index >= self.pages.count():
            return
        self.pages.setCurrentIndex(index)
        self.review_tab.setChecked(index == 0)
        self.reconstruction_tab.setChecked(index == 1)
        self.labeling_tab.setChecked(index == 2)
        if index == 1:
            self.reconstruction_page.refresh_sources()
        elif index == 2:
            self.labeling_page.refresh_sources()

    def refresh_analysis_sources(self, preferred=None):
        if hasattr(self, 'reconstruction_page'):
            self.reconstruction_page.refresh_sources(preferred)
        if hasattr(self, 'labeling_page'):
            self.labeling_page.refresh_sources()

    def refresh_overlay_sources(self):
        for pane in self.panes[:3]:
            pane.set_overlay_sources(self.datasets.values())

    def add_generated_dataset(self, dataset):
        old = next((d for d in self.datasets.values() if d.paths == dataset.paths
                    and d.domain == dataset.domain and d.mask == dataset.mask), None)
        if old is not None:
            dataset.close()
            dataset = old
        else:
            self.datasets[dataset.id] = dataset
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, dataset.id)
            self.dataset_list.addItem(item)
            self.update_item(dataset)
        for row in range(self.dataset_list.count()):
            item = self.dataset_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == dataset.id:
                self.dataset_list.setCurrentItem(item)
                break
        self.refresh_analysis_sources()
        self.refresh_overlay_sources()
        self.statusBar().showMessage(f'已暂存到图像管理：{dataset.name}', 8000)

    def options(self):
        return ImportOptions(self.import_domain.currentData(), self.import_mask.isChecked(),
                             self.raw_height.value(), self.raw_width.value(),
                             self.raw_dtype.currentData(), self.use_xml.isChecked(),
                             self.recon_height.value(), self.recon_width.value(),
                             self.auto_recon_shape.isChecked())

    def selected_dataset(self):
        item = self.dataset_list.currentItem()
        return self.datasets.get(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def select_pane(self, index):
        self.active_pane = index
        for pane in self.panes:
            pane.set_active(pane.index == index)
        self.update_percentile_actions()
        if hasattr(self, 'overlay_stack'):
            self.overlay_stack.setCurrentIndex(index)
            self.overlay_target.setText(f'画布 {index+1:02d} · 投影叠加' if index < 3 else '当前为重建幕布')
            self.sync_overlay_tools()
        dataset = self.panes[index].dataset
        if dataset is not None:
            for row in range(self.dataset_list.count()):
                item = self.dataset_list.item(row)
                if item.data(Qt.ItemDataRole.UserRole) == dataset.id:
                    self.dataset_list.setCurrentItem(item)
                    break

    def sync_overlay_tools(self, *_):
        for pane in self.panes[:3]:
            with QSignalBlocker(pane.overlay_toggle):
                pane.overlay_toggle.setChecked(pane.index == self.active_pane and self.overlay_tools_toggle.isChecked())

    def show_overlay_tools(self, index, expanded):
        if not hasattr(self, 'overlay_tools_toggle'):
            return
        self.select_pane(index)
        self.overlay_tools_toggle.setChecked(expanded)
        self.sync_overlay_tools()
        if expanded:
            self.tools_sidebar.ensureWidgetVisible(self.overlay_tools_toggle)

    def toggle_pane_maximize(self, index):
        self.maximized_pane = None if self.maximized_pane == index else index
        self.arrange_panes()
        if self.maximized_pane is not None:
            self.select_pane(index)

    def restore_panes(self):
        if self.pages.currentIndex() == 2:
            self.labeling_page.cancel_gestures()
            return
        if self.pages.currentIndex() == 1:
            self.reconstruction_page.set_volume_maximized(False)
            return
        if self.maximized_pane is not None:
            self.maximized_pane = None
            self.arrange_panes()

    def arrange_panes(self):
        for pane in self.panes:
            self.board_grid.removeWidget(pane)
        for pane in self.panes:
            maximized = pane.index == self.maximized_pane
            pane.maximize_button.setChecked(maximized)
            pane.maximize_button.setText('还原' if maximized else '放大')
            if self.maximized_pane is None:
                self.board_grid.addWidget(pane, pane.index // 2, pane.index % 2)
                pane.show()
            elif maximized:
                self.board_grid.addWidget(pane, 0, 0, 2, 2)
                pane.show()
            else:
                pane.hide()

    def receive_drop(self, index, mime):
        self.select_pane(index)
        if mime.hasFormat(MIME_DATASET):
            dataset = self.datasets.get(bytes(mime.data(MIME_DATASET)).decode())
            if dataset:
                self.place(dataset, index)
        else:
            self.import_paths(local_paths(mime), index)

    def place_overlay(self, dataset, index):
        self.select_pane(index)
        try:
            self.panes[index].set_overlay(dataset)
        except ValueError as error:
            self.statusBar().showMessage(str(error), 12000)
            return False
        self.statusBar().showMessage(f'画布 {index+1:02d} 已叠加 {dataset.name}；按真实帧号同步，可调整颜色和不透明度', 8000)
        return True

    def overlay_source_changed(self, index, dataset_id):
        pane = self.panes[index]
        if dataset_id is None:
            pane.clear_overlay()
        else:
            dataset = self.datasets.get(dataset_id)
            if dataset is None or not self.place_overlay(dataset, index):
                pane.sync_overlay_source()

    def dragEnterEvent(self, event):
        if self.pages.currentIndex() == 1:
            event.ignore()
            return
        if local_paths(event.mimeData()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self.pages.currentIndex() == 1:
            event.ignore()
            return
        paths = local_paths(event.mimeData())
        if paths:
            self.import_paths(paths)
            event.acceptProposedAction()

    def open_files(self):
        if self.pages.currentIndex() == 1:
            self.reconstruction_page.open_files()
            return
        paths, _ = QFileDialog.getOpenFileNames(self, '添加当前样本的图像', '',
            '支持的图像 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.npy *.prj *.slc *.bin *.raw)')
        if paths:
            self.import_paths(paths)

    def open_folder(self):
        if self.pages.currentIndex() == 1:
            self.reconstruction_page.open_folder()
            return
        path = QFileDialog.getExistingDirectory(self, '添加一个图像序列文件夹')
        if path:
            self.import_paths([path])

    def import_paths(self, paths, target=-1, analysis_dtype=None):
        if not paths:
            return
        options = self.options()
        if target == -1 and self.pages.currentIndex() == 1:
            target = -2
        if target == -1 and self.pages.currentIndex() == 2:
            target = -3
        if target == -3:
            options = replace(options, domain='reconstruction', mask=False)
        if target == -2:
            options = replace(options, domain='projection', mask=True)
            if analysis_dtype is not None:
                options = replace(options, dtype=analysis_dtype)
        if target >= 0:
            options = replace(options, domain='reconstruction' if target == 3 else 'projection')
        job = LoadJob(self.epoch, target, paths, options)
        self.jobs.append(job)
        job.signals.finished.connect(lambda epoch, dst, data, errors, j=job:
                                     self.import_finished(j, epoch, dst, data, errors))
        self.statusBar().showMessage('正在读取数据… 可继续操作或一键清空')
        QThreadPool.globalInstance().start(job)

    def import_finished(self, job, epoch, target, datasets, errors):
        if job in self.jobs:
            self.jobs.remove(job)
        if epoch != self.epoch:
            for dataset in datasets:
                dataset.close()
            return
        added = []
        for dataset in datasets:
            duplicate = next((old for old in self.datasets.values()
                              if old.paths == dataset.paths and old.domain == dataset.domain
                              and old.mask == dataset.mask and old.dtype == dataset.dtype
                              and old.shape == dataset.shape), None)
            if duplicate:
                dataset.close()
                added.append(duplicate)
                continue
            self.datasets[dataset.id] = dataset
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, dataset.id)
            self.dataset_list.addItem(item)
            self.update_item(dataset)
            self.dataset_list.setCurrentItem(item)
            added.append(dataset)
        for i, dataset in enumerate(added):
            if target in (-2, -3):
                continue
            if i == 0 and target >= 0 and (target == 3) == (dataset.domain == 'reconstruction'):
                self.place(dataset, target)
            else:
                empty = next((p.index for p in self.panes if p.dataset is None and
                              (p.index == 3) == (dataset.domain == 'reconstruction')), None)
                if empty is not None:
                    self.place(dataset, empty)
        self.statusBar().showMessage(f'已加载 {len(self.datasets)} 组数据' +
            (' · 还有导入任务进行中' if self.jobs else ' · 从左侧拖到画布即可比较'))
        if errors:
            QMessageBox.warning(self, '部分数据未能加载', '\n\n'.join(errors[:8]))
        self.refresh_analysis_sources(added[0].id if target == -2 and added else None)
        self.refresh_overlay_sources()

    def update_item(self, dataset):
        for i in range(self.dataset_list.count()):
            item = self.dataset_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == dataset.id:
                domain = '投影' if dataset.domain == 'projection' else '重建'
                mode = (' · 掩膜' if dataset.mask else '') + (' · 暂存' if dataset.generated else '')
                panes = [p.index + 1 for p in self.panes if p.dataset is dataset or p.overlay_dataset is dataset]
                item.setData(DISPLAY_PANES_ROLE, panes)
                locations = [f'{p.index+1:02d}' + ('（叠加）' if p.overlay_dataset is dataset and p.dataset is not dataset else '')
                             for p in self.panes if p.dataset is dataset or p.overlay_dataset is dataset]
                showing = '\n显示于画布 ' + ' / '.join(locations) if panes else ''
                item.setText(f'{dataset.name}\n{domain}{mode}  ·  {len(dataset.indices)} 层  ·  {dataset.dtype}{showing}')
                item.setToolTip((dataset.description + '\n' if dataset.description else '') +
                                '\n'.join(str(p) for p in dataset.geometry_paths[:4]))
                break

    def update_displayed_items(self):
        if hasattr(self, 'dataset_list'):
            for dataset in self.datasets.values():
                self.update_item(dataset)

    def place(self, dataset, index):
        if (index == 3) != (dataset.domain == 'reconstruction'):
            self.statusBar().showMessage('数据域不匹配：投影放前三窗，重建放右下窗；可在左侧修改数据域', 8000)
            return
        pane = self.panes[index]
        number = self.link_number if index < 3 and self.link.isChecked() and pane.bind.isChecked() else None
        peers = [p for p in self.bound_panes() if p.index != index] if number is not None else []
        # The first bound sequence establishes its own numbering (e.g. 1-based files).
        other_bound = bool(peers)
        if number is not None and not other_bound:
            number = dataset.indices[0]
            self.link_number = number
        if peers:
            pane.view.set_markers(peers[0].view.crosshair, peers[0].view.selection)
        pane.assign(dataset, number)
        if peers:
            pane.view.apply_view_state(peers[0].view.view_state())
        self.select_pane(index)
        self.update_link_range()

    def place_selected(self, *_):
        dataset = self.selected_dataset()
        if dataset:
            self.place(dataset, self.active_pane)

    def show_details(self, *_):
        dataset = self.selected_dataset()
        self._updating_details = True
        self.domain.setEnabled(dataset is not None and not dataset.generated)
        self.mask.setEnabled(dataset is not None and not dataset.generated)
        if dataset:
            ids = dataset.indices
            self.details.setText(f'{dataset.name}\n尺寸：{" × ".join(map(str, dataset.shape))}\n'
                f'层号：{min(ids)}–{max(ids)} · 共 {len(ids)} 层\n'
                f'{"暂存处理结果 · 继承真实层号" if dataset.generated else "文件名层号" if dataset.numbered else "从 0 开始的位置编号"}\n{dataset.shape_source}')
            self.details.setToolTip((dataset.description + '\n' if dataset.description else '') +
                                   str(dataset.geometry_paths[0]))
            self.domain.setCurrentIndex(0 if dataset.domain == 'projection' else 1)
            self.mask.setChecked(dataset.mask)
        else:
            self.details.setText('暂无选中数据')
            self.details.setToolTip('')
            self.mask.setChecked(False)
        self._updating_details = False

    def edit_domain(self, *_):
        dataset = self.selected_dataset()
        if self._updating_details or dataset is None:
            return
        dataset.domain = self.domain.currentData()
        for pane in self.panes:
            if pane.dataset is dataset:
                pane.clear()
            elif pane.overlay_dataset is dataset:
                pane.clear_overlay()
        self.update_item(dataset)
        self.show_details()
        self.refresh_analysis_sources()
        self.refresh_overlay_sources()
        self.statusBar().showMessage('数据域已更新，可拖入对应画布。Raw 尺寸和 dtype 的修改需要移除后重新导入', 8000)

    def edit_mask(self, checked):
        dataset = self.selected_dataset()
        if self._updating_details or dataset is None:
            return
        dataset.mask = checked
        self.update_item(dataset)
        for pane in self.panes:
            if pane.dataset is dataset:
                pane.set_layer(pane.number)
            if pane.overlay_dataset is dataset and not checked:
                pane.clear_overlay()
        self.refresh_analysis_sources()
        self.refresh_overlay_sources()
        self.update_percentile_actions()

    def remove_selected(self):
        dataset = self.selected_dataset()
        if dataset is None:
            return
        if not self.confirm_unsaved_labels(dataset.id):
            return
        for pane in self.panes:
            if pane.dataset is dataset:
                pane.clear()
            elif pane.overlay_dataset is dataset:
                pane.clear_overlay()
        self.datasets.pop(dataset.id)
        if self._percentile_cached_id == dataset.id:
            self._percentile_cached_id = self._percentile_statistics = None
        self.dataset_list.takeItem(self.dataset_list.currentRow())
        dataset.close()
        self.show_details()
        self.statusBar().showMessage(f'已移除数据，剩余 {len(self.datasets)} 组')
        self.refresh_analysis_sources()
        self.refresh_overlay_sources()

    def bound_panes(self):
        return [p for p in self.panes[:3] if p.dataset is not None and p.bind.isChecked()]

    def update_link_range(self):
        if not hasattr(self, 'link_slider'):
            return
        panes = self.bound_panes()
        self.link_slider.setEnabled(bool(panes) and self.link.isChecked())
        with QSignalBlocker(self.link_slider):
            self.link_slider.setRange(min((min(p.dataset.indices) for p in panes), default=0),
                                      max((max(p.dataset.indices) for p in panes), default=0))
            self.link_slider.setValue(self.link_number)
        self.link_label.setText(f'层号 {self.link_number}' if panes else '未加载投影')

    def change_link_layer(self, number):
        self.link_number = number
        for pane in self.bound_panes():
            pane.set_layer(number)
        self.update_link_range()
        self.refresh_reference_points()

    def change_layer(self, index, number):
        pane = self.panes[index]
        if index < 3 and self.link.isChecked() and pane.bind.isChecked():
            self.change_link_layer(number)
        else:
            pane.set_layer(number)
            self.refresh_reference_points()

    def toggle_link(self, checked):
        if checked:
            panes = self.bound_panes()
            source = self.panes[self.active_pane]
            if source not in panes:
                source = panes[0] if panes else None
            if source:
                self.change_link_layer(source.number)
                self.sync_view(source.index, source.view.view_state())
                self.sync_markers(source.index, source.view.crosshair, source.view.selection)
        self.update_link_range()
        self.refresh_reference_points()

    def align_binding(self, pane, checked):
        if checked and pane.dataset and self.link.isChecked():
            pane.set_layer(self.link_number)
            peers = [p for p in self.bound_panes() if p is not pane]
            if peers:
                pane.view.set_markers(peers[0].view.crosshair, peers[0].view.selection)
                pane.view.apply_view_state(peers[0].view.view_state())
        self.update_link_range()
        self.refresh_reference_points()

    def fit_all(self):
        if self.pages.currentIndex() == 2:
            self.labeling_page.fit_all()
            return
        if self.pages.currentIndex() == 1:
            self.reconstruction_page.fit_all()
            return
        for pane in self.panes:
            for view in pane.section_views:
                view.fit_image()

    def sync_orthogonal_position(self, index, normalized):
        source = self.panes[index]
        if index == 3 or not self.link.isChecked() or not source.bind.isChecked():
            return
        for pane in self.bound_panes():
            if pane is not source:
                pane.set_orthogonal_position(normalized, notify=False)

    def sync_auxiliary_view(self, index, axis, state):
        source = self.panes[index]
        if index == 3 or not self.link.isChecked() or not source.bind.isChecked():
            return
        for pane in self.bound_panes():
            if pane is not source:
                pane.section_views[axis].apply_view_state(state)

    def sync_markers(self, index, crosshair, selection):
        source = self.panes[index]
        if index == 3 or not self.link.isChecked() or not source.bind.isChecked():
            return
        for pane in self.bound_panes():
            pane.view.set_markers(crosshair, selection)

    def sync_view(self, index, state):
        source = self.panes[index]
        if index == 3 or not self.link.isChecked() or not source.bind.isChecked():
            return
        for pane in self.bound_panes():
            if pane is not source:
                pane.view.apply_view_state(state)

    def pick_reconstruction_point(self, normalized):
        pane = self.panes[3]
        if not self.geometry_enabled.isChecked() or pane.dataset is None or pane._frame is None:
            return
        # Clear a previous mapping before loading geometry so a failed new
        # pick can never leave another anatomical point highlighted in red.
        self.reference = None
        self.ray_reference = None
        try:
            dataset = pane.dataset
            geometry = self._geometry_cache.get(dataset.id)
            if geometry is None:
                geometry = ProjectionGeometry.from_dataset(dataset)
                self._geometry_cache[dataset.id] = geometry
            voxel, world = geometry.voxel(normalized, pane.number)
            self.reference = (dataset, geometry, voxel, world)
            self.clear_ordinary_crosshairs()
        except GeometryError as error:
            self.refresh_reference_points()
            self.geometry_info.setText(f'几何定位未建立：{error}')
            self.geometry_info.setToolTip(str(error))
            self.statusBar().showMessage(str(error), 10000)
            return
        self.refresh_reference_points()

    def pick_projection_ray(self, index, normalized):
        source, recon = self.panes[index], self.panes[3]
        if (not self.geometry_enabled.isChecked() or source.dataset is None or
                source._frame is None or recon.dataset is None):
            return
        self.reference = None
        self.ray_reference = None
        try:
            geometry = self._geometry_cache.get(recon.dataset.id)
            if geometry is None:
                geometry = ProjectionGeometry.from_dataset(recon.dataset)
                self._geometry_cache[recon.dataset.id] = geometry
            geometry.check_projection_dataset(source.dataset)
            geometry.backproject(normalized, source.number, source._frame.shape)
            self.ray_reference = (index, source.dataset, recon.dataset, geometry, normalized)
            self.clear_ordinary_crosshairs()
        except GeometryError as error:
            self.refresh_reference_points()
            self.geometry_info.setText(f'射线定位未建立：{error}')
            self.geometry_info.setToolTip(str(error))
            self.statusBar().showMessage(str(error), 10000)
            return
        self.refresh_reference_points()

    def refresh_reference_points(self, *_):
        if not hasattr(self, 'geometry_info'):
            return
        for pane in self.panes:
            pane.view.set_reference_point()
            pane.view.set_ray_trace()
        if self.ray_reference is not None:
            self.refresh_ray_reference()
            return
        if self.reference is None:
            self.geometry_info.setText('点击重建：投影对应点\n点击投影：重建射线 / 切片交点')
            self.geometry_info.setToolTip('')
            return
        source, geometry, voxel, world = self.reference
        if self.panes[3].dataset is not source:
            self.reference = None
            self.refresh_reference_points()
            return
        if not self.geometry_enabled.isChecked():
            self.geometry_info.setText('重建 → 投影定位已关闭')
            return
        col, row, layer = voxel
        recon = self.panes[3]
        if recon.number == layer and recon._frame is not None:
            height, width = geometry.slice_shape
            recon.view.set_reference_point(((col + .5) / width, (row + .5) / height))
        notes, coordinates = [], []
        for pane in self.panes[:3]:
            if pane.dataset is None:
                continue
            if pane._frame is None:
                notes.append(f'{pane.index + 1:02d}：当前层缺失')
                continue
            try:
                geometry.check_projection_dataset(pane.dataset)
                point = geometry.project(world, pane.number, pane._frame.shape)
                pane.view.set_reference_point(point)
                h, w = pane._frame.shape[:2]
                coordinates.append(f'画布 {pane.index + 1:02d} 红十字 [N,Y,X]：'
                                   f'[{pane.number}, {point[1] * h - .5:.2f}, {point[0] * w - .5:.2f}]')
            except GeometryError as error:
                notes.append(f'{pane.index + 1:02d}：{error}')
        summary = f'重建点 ({col}, {row}, 层 {layer})'
        if notes:
            summary += '\n' + '\n'.join(notes)
        else:
            summary += '\n红十字随投影角度更新'
        self.geometry_info.setText(summary)
        self.geometry_info.setToolTip(
            f'世界坐标 XYZ (mm)：{world[0]:.3f}, {world[1]:.3f}, {world[2]:.3f}\n'
            f'坐标转换：{geometry.coordinate_convention}\n几何目录：{geometry.root}\n' +
            '\n'.join(coordinates + notes))

    def clear_ordinary_crosshairs(self):
        for pane in self.panes:
            pane.view.set_navigation_point()
            for view in pane.section_views:
                view.set_markers(None, view.selection)

    def clear_position_overlays(self):
        self.reference = None
        self.ray_reference = None
        self.clear_ordinary_crosshairs()
        self.refresh_reference_points()

    def refresh_ray_reference(self):
        index, dataset, recon_dataset, geometry, normalized = self.ray_reference
        source, recon = self.panes[index], self.panes[3]
        if source.dataset is not dataset or recon.dataset is not recon_dataset:
            self.ray_reference = None
            self.refresh_reference_points()
            return
        if not self.geometry_enabled.isChecked():
            self.geometry_info.setText('几何定位已关闭')
            return
        if source._frame is None:
            self.geometry_info.setText('射线来源投影当前层缺失，暂不绘制')
            return
        try:
            ray = geometry.backproject(normalized, source.number, source._frame.shape)
            for pane in self.panes[:3]:
                if pane.dataset is not None and pane._frame is not None and pane.number == source.number:
                    try:
                        geometry.check_projection_dataset(pane.dataset)
                        geometry.project(ray.world_endpoints.mean(axis=0), pane.number, pane._frame.shape)
                        pane.view.set_reference_point(normalized)
                    except GeometryError:
                        pass
            point, line = ray.section(recon.number)
            if recon._frame is not None:
                recon.view.set_ray_trace(ray.top_view, line)
                recon.view.set_reference_point(point)
            section = ('当前切片：射线共面，显示红色实线' if line is not None else
                       '当前切片：红十字为真实交点' if point is not None else
                       '当前切片：无交点（橙虚线仅为俯视轨迹）')
            if recon._frame is None:
                section = '当前重建层缺失，暂不绘制'
            self.geometry_info.setText(f'射线来源：画布 {index + 1:02d} · 投影 {source.number}\n'
                                       f'橙虚线：射线俯视轨迹\n{section}')
            self.geometry_info.setToolTip(
                f'橙色虚线是体内三维射线的俯视轨迹，并非切片内的实际线。\n'
                f'坐标转换：{geometry.coordinate_convention}\n'
                f'体内射线端点 XYZ (mm)：{ray.world_endpoints[0].round(3).tolist()} → '
                f'{ray.world_endpoints[1].round(3).tolist()}\n'
                f'射线随来源画布的投影角度更新；沿重建切片浏览时显示各平面的交点。')
        except GeometryError as error:
            self.geometry_info.setText(f'当前射线不可见：{error}')
            self.geometry_info.setToolTip(str(error))

    def clear_markers(self):
        self.reference = None
        self.ray_reference = None
        for pane in self.panes:
            pane.view.set_markers()
            pane.view.set_reference_point()
            pane.view.set_ray_trace()
        self.refresh_reference_points()

    def clear_all(self):
        self.cancel_percentile_window(notify=False)
        self._percentile_cached_id = self._percentile_statistics = None
        self._percentile_frame_key = self._percentile_frame_statistics = None
        if hasattr(self, 'reconstruction_page'):
            self.reconstruction_page.clear()
        if hasattr(self, 'labeling_page'):
            self.labeling_page.clear()
        self.epoch += 1
        self.reference = None
        self.ray_reference = None
        self._geometry_cache.clear()
        for job in self.jobs:
            job.cancelled.set()
        for pane in self.panes:
            pane.clear()
        for dataset in self.datasets.values():
            dataset.close()
        self.datasets.clear()
        self.dataset_list.clear()
        self.link_number = 0
        self.clear_markers()
        self.update_link_range()
        self.show_details()
        self.refresh_analysis_sources()
        self.refresh_overlay_sources()
        self.statusBar().showMessage('当前样本已清空，可拖入下一个样本')

    def confirm_unsaved_labels(self, dataset_id=None):
        page = getattr(self, 'labeling_page', None)
        if page is None or not any(session.revision != session.saved_revision
                                   for key, session in page.sessions.items()
                                   if dataset_id is None or key == dataset_id):
            return True
        return QMessageBox.question(self, '尚有未保存的标签',
            '标签勾画中有未保存的修改。请取消并保存标签文件，或选择放弃后继续。',
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel) == QMessageBox.StandardButton.Discard

    def request_clear_all(self):
        if self.confirm_unsaved_labels():
            self.clear_all()

    def closeEvent(self, event):
        if not self.confirm_unsaved_labels():
            event.ignore()
            return
        self._startup_placement_timer.stop()
        self.clear_all()
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName('数据审查')
    app.setStyle('Fusion')
    window = ReviewWindow()
    window.show()
    if len(sys.argv) > 1:
        window.import_paths(sys.argv[1:])
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
