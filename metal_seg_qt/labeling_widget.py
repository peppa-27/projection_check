"""ITK-SNAP-inspired label workspace: three slices share one editable volume."""
from threading import Event

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, Qt, QThreadPool, Signal, QPointF
from PySide6.QtGui import QColor, QKeySequence, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QFormLayout, QGroupBox, QScrollArea,
    QSplitter, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog,
    QColorDialog, QAbstractItemView, QTabWidget, QSizePolicy, QButtonGroup, QRadioButton)

from .app import ImageView, button, MIME_DATASET, local_paths
from .data import auto_window, display_pixels
from .geometry import ProjectionGeometry
from .processing import ProcessingSource, ProcessingCancelled, project_mask
from .labeling_core import (Label, LabelSession, plane, brush_selection, polygon_selection)


class TaskSignals(QObject):
    progress = Signal(str)
    finished = Signal(object, object)


class LabelTask(QRunnable):
    def __init__(self, source_id, work):
        super().__init__()
        self.source_id, self.work = source_id, work
        self.cancelled = Event()
        self.signals = TaskSignals()

    def run(self):
        result, error = None, None
        try:
            result = self.work(self.cancelled, self.signals.progress.emit)
        except ProcessingCancelled:
            pass
        except Exception as caught:
            error = str(caught)
        finally:
            self.work = None
        self.signals.finished.emit(result, error)


class LabelView(ImageView):
    def __init__(self, workspace, axis):
        super().__init__()
        self.workspace, self.axis = workspace, axis
        self.points = []
        self.stroke = None
        self.vertex = None
        self.vertex_press = None
        self.hover = None
        self.setMouseTracking(True)
        self.setAcceptDrops(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.hint = '从左侧选择重建数据，点击「打开 / 继续勾画」'
        self.stepped.connect(lambda step: workspace.step(axis, step))
        self.windowDragged.connect(workspace.window_dragged)

    def pixel_position(self, event):
        position = self.image_position(event.position())
        if position is None:
            return None
        bounds = self.pixmap_item.boundingRect()
        return (min(bounds.width()-.5, position[0]*bounds.width()),
                min(bounds.height()-.5, position[1]*bounds.height()))

    def mousePressEvent(self, event):
        work = self.workspace
        if work.busy or work.session is None:
            event.accept()
            return
        work.active_axis = self.axis
        self.setFocus()
        point = self.pixel_position(event)
        tool = work.tool_key
        if point is not None and event.button() == Qt.MouseButton.LeftButton and tool == 'navigate':
            work.navigate(self.axis, point)
            event.accept()
        elif point is not None and tool == 'polygon' and event.button() == Qt.MouseButton.LeftButton:
            tolerance = 7 / max(.01, self.transform().m11())
            self.vertex = next((i for i, p in enumerate(self.points)
                if np.hypot(p[0]-point[0], p[1]-point[1]) <= tolerance), None)
            self.vertex_press = event.position() if self.vertex is not None else None
            if self.vertex is None:
                self.points.append(point)
            self.viewport().update()
            work.status.setText('点击首点 / 双击 / 右键 / Enter 闭合；拖动顶点修正；Esc 放弃')
            event.accept()
        elif tool == 'polygon' and event.button() == Qt.MouseButton.RightButton:
            work.accept_polygon(axis=self.axis)
            event.accept()
        elif point is not None and tool in ('brush', 'erase') and event.button() in (
                Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            index = work.position[self.axis]
            before = plane(work.session.labels, self.axis, index).copy()
            erase = tool == 'erase' or event.button() == Qt.MouseButton.RightButton
            self.stroke = (index, before, point, erase, work.active_label(), work.paint_policy,
                           work.brush_size.value()/2)
            self.paint_to(point)
            event.accept()
        else:
            super().mousePressEvent(event)

    def paint_to(self, point):
        index, before, previous, erase, active, policy, radius = self.stroke
        target = plane(self.workspace.session.labels, self.axis, index)
        selected = brush_selection(target.shape, previous, point, radius)
        self.workspace.session.paint(self.axis, index, selected, active, policy, erase)
        self.stroke = (index, before, point, erase, active, policy, radius)
        self.workspace.render()

    def mouseMoveEvent(self, event):
        self.hover = self.pixel_position(event)
        if self.vertex is not None and self.hover is not None:
            self.points[self.vertex] = self.hover
            event.accept()
        elif self.stroke is not None:
            if self.hover is not None:
                self.paint_to(self.hover)
            event.accept()
        else:
            super().mouseMoveEvent(event)
        self.viewport().update()

    def mouseReleaseEvent(self, event):
        if self.vertex is not None:
            close = (self.vertex == 0 and len(self.points) >= 3 and self.vertex_press is not None
                     and (event.position()-self.vertex_press).manhattanLength() <= 7)
            self.vertex = None
            self.vertex_press = None
            if close:
                self.workspace.accept_polygon(axis=self.axis)
            event.accept()
        elif self.stroke is not None and event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            point = self.pixel_position(event)
            if point is not None:
                self.paint_to(point)
            index, before, _, erase, *_ = self.stroke
            self.stroke = None
            self.workspace.session.commit_plane(self.axis, index, before, '橡皮' if erase else '画笔')
            self.workspace.changed()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if (self.workspace.tool_key == 'polygon' and event.button() == Qt.MouseButton.LeftButton
                and not self.workspace.busy and len(self.points) >= 3):
            self.vertex = self.vertex_press = None
            self.workspace.accept_polygon(axis=self.axis)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        if self.stroke is not None or self.workspace.busy:
            event.accept()
        else:
            super().wheelEvent(event)

    def leaveEvent(self, event):
        self.hover = None
        self.viewport().update()
        super().leaveEvent(event)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not self.has_image():
            return
        pen = QPen(QColor('#ffdd76'), 1.5)
        pen.setCosmetic(True)
        painter.setPen(pen)
        if self.points:
            path = QPainterPath(QPointF(*self.points[0]))
            for point in self.points[1:]:
                path.lineTo(QPointF(*point))
            if self.hover is not None:
                path.lineTo(QPointF(*self.hover))
            painter.drawPath(path)
            radius = 4 / max(.01, self.transform().m11())
            for i, (x, y) in enumerate(self.points):
                painter.setPen(QPen(QColor('#65e2d3') if i == 0 and len(self.points) >= 3
                                    else QColor('#ffdd76'), 0))
                painter.drawEllipse(QPointF(x, y), radius, radius)
        elif self.hover is not None and self.workspace.tool_key in ('brush', 'erase'):
            radius = self.workspace.brush_size.value()/2
            painter.drawEllipse(QPointF(*self.hover), radius, radius)


class LabelingWidget(QWidget):
    datasetGenerated = Signal(object)

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.sessions = {}
        self.session = None
        self.source_id = None
        self.position = [0, 0, 0]
        self.active_axis = 0
        self.tool_key = 'navigate'
        self.paint_policy = 'background'
        self.last_polygons = {}
        self.projection_ready = False
        self.job = None
        self.epoch = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)
        title = QLabel('标签勾画  /  重建 → 标签 → 投影')
        title.setObjectName('heading')
        layout.addWidget(title)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(split, 1)
        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 0)
        split.addWidget(sidebar)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(330)
        controls = QWidget()
        left = QVBoxLayout(controls)
        scroll.setWidget(controls)
        side_layout.addWidget(scroll, 1)
        source_group = QGroupBox('1 · 重建数据')
        source_form = QVBoxLayout(source_group)
        self.source = QComboBox()
        self.source.setMinimumWidth(0)
        self.source.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.source.setMinimumContentsLength(12)
        source_form.addWidget(self.source)
        self.open_button = button('打开 / 继续勾画', self.open_source)
        source_form.addWidget(self.open_button)
        self.source_info = QLabel('先在数据审查中导入重建数据。切换数据保留本次会话的标签。')
        self.source_info.setWordWrap(True)
        source_form.addWidget(self.source_info)
        left.addWidget(source_group)
        self.editor = QWidget()
        editor_layout = QVBoxLayout(self.editor)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        left.addWidget(self.editor)
        label_group = QGroupBox('2 · 标签管理')
        labels_layout = QVBoxLayout(label_group)
        self.label_table = QTableWidget(0, 5)
        self.label_table.setHorizontalHeaderLabels(['ID', '名称 / 颜色', '显示', '锁定', '输出'])
        self.label_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.label_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column, width in ((0, 28), (2, 34), (3, 34), (4, 34)):
            self.label_table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self.label_table.setColumnWidth(column, width)
        self.label_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.label_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.label_table.verticalHeader().hide()
        self.label_table.setFixedHeight(58)
        self.label_table.itemChanged.connect(self.edit_label)
        self.label_table.itemSelectionChanged.connect(self.render)
        self.label_table.cellDoubleClicked.connect(self.pick_color)
        labels_layout.addWidget(self.label_table)
        hint = QLabel('默认勾画金属；需要时再添加标签。')
        hint.setWordWrap(True)
        hint.setToolTip('点击行选为当前标签；名称可编辑，双击 ID 改颜色。0 为背景；锁定标签受保护。')
        label_row = QHBoxLayout()
        label_row.addWidget(hint, 1)
        label_row.addWidget(button('+ 标签', self.add_label))
        labels_layout.addLayout(label_row)
        editor_layout.addWidget(label_group)
        self.tool_tabs = QTabWidget()
        editor_layout.addWidget(self.tool_tabs)
        tools = QGroupBox('3 · 勾画工具')
        tools.setStyleSheet('QPushButton#labelTool:checked { background:#245956; '
                           'border:1px solid #63d3c7; color:#e4fff8; font-weight:600; }')
        form = QFormLayout(tools)
        self.tool_group = QButtonGroup(self)
        self.tool_buttons = {}
        tool_row = QHBoxLayout()
        for name, key in [('定位', 'navigate'), ('画笔', 'brush'), ('橡皮', 'erase'), ('多边形', 'polygon')]:
            tool_button = button(name, lambda checked=False, k=key: self.set_tool(k))
            tool_button.setObjectName('labelTool')
            tool_button.setCheckable(True)
            self.tool_group.addButton(tool_button)
            self.tool_buttons[key] = tool_button
            tool_row.addWidget(tool_button)
        self.tool_buttons['navigate'].setChecked(True)
        form.addRow(tool_row)
        self.brush_size = QSpinBox()
        self.brush_size.setRange(1, 200)
        self.brush_size.setValue(10)
        self.brush_size.setSuffix(' 像素直径')
        form.addRow('二维画笔', self.brush_size)
        self.policy_group = QButtonGroup(self)
        policy_row = QHBoxLayout()
        for name, key in [('保护其他标签', 'background'), ('覆盖未锁定标签', 'all')]:
            choice = QRadioButton(name)
            choice.setChecked(key == self.paint_policy)
            choice.toggled.connect(lambda checked, k=key: self.set_policy(k) if checked else None)
            self.policy_group.addButton(choice)
            policy_row.addWidget(choice)
        form.addRow(policy_row)
        row = QHBoxLayout()
        self.accept_polygon_button = button('闭合并填充 (Enter)', lambda: self.accept_polygon())
        row.addWidget(self.accept_polygon_button)
        row.addWidget(button('放弃 (Esc)', self.cancel_gestures))
        form.addRow(row)
        form.addRow(button('粘贴上次轮廓 (Ctrl+V)', self.paste_polygon))
        row = QHBoxLayout()
        self.undo_button = button('撤销', lambda: self.history(False))
        self.redo_button = button('重做', lambda: self.history(True))
        row.addWidget(self.undo_button)
        row.addWidget(self.redo_button)
        form.addRow(row)
        self.show_labels = QCheckBox('显示标签')
        self.show_labels.setChecked(True)
        self.show_labels.toggled.connect(self.render)
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.valueChanged.connect(self.render)
        form.addRow(self.show_labels)
        form.addRow('标签不透明度', self.opacity)
        self.tool_tabs.addTab(tools, '手工勾画')
        thresholds = QGroupBox('阈值辅助')
        form = QFormLayout(thresholds)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(-1e12, 1e12)
        self.threshold.setDecimals(4)
        self.threshold.setValue(3000)
        self.threshold.setMinimumWidth(0)
        self.threshold.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.comparison = QComboBox()
        self.comparison.addItem('原始值 ≥', 'ge')
        self.comparison.addItem('原始值 ≤', 'le')
        row = QHBoxLayout()
        row.addWidget(self.comparison)
        row.addWidget(self.threshold)
        form.addRow(row)
        self.threshold_preview = QCheckBox('黄色预览（不修改标签）')
        for signal in (self.threshold.valueChanged, self.comparison.currentIndexChanged,
                       self.threshold_preview.toggled):
            signal.connect(self.render)
        form.addRow(self.threshold_preview)
        row = QHBoxLayout()
        row.addWidget(button('应用当前切面', lambda: self.apply_threshold(False)))
        row.addWidget(button('应用整组', lambda: self.apply_threshold(True)))
        form.addRow(row)
        self.tool_tabs.addTab(thresholds, '阈值辅助')
        output = QGroupBox('4 · 保存与生成')
        form = QVBoxLayout(output)
        row = QHBoxLayout()
        row.addWidget(button('保存标签文件', self.save_labels))
        row.addWidget(button('载入标签文件', self.load_labels))
        form.addLayout(row)
        self.output_scope = QComboBox()
        self.output_scope.addItem('当前标签', 'active')
        self.output_scope.addItem('勾选标签合并', 'selected')
        form.addWidget(self.output_scope)
        self.export_button = button('标签 → 重建掩膜（图像管理）', self.export_labels)
        self.project_button = button('标签 → 投影域掩膜（全部角度）', self.project_labels)
        form.addWidget(self.export_button)
        form.addWidget(self.project_button)
        self.geometry_info = QLabel('')
        self.geometry_info.setWordWrap(True)
        form.addWidget(self.geometry_info)
        self.output_panel = output
        side_layout.addWidget(output)
        self.geometry_info.setMaximumHeight(65)
        left.addStretch()
        self.cancel_button = button('取消后台任务', self.cancel_task)
        side_layout.addWidget(self.cancel_button)
        self.status = QLabel('选择重建数据后开始；标签文件需主动保存，图像管理中的结果为会话暂存。')
        self.status.setWordWrap(True)
        side_layout.addWidget(self.status)
        board = QWidget()
        board_layout = QVBoxLayout(board)
        board_layout.setContentsMargins(5, 0, 0, 0)
        window_row = QHBoxLayout()
        self.ww, self.wl = QDoubleSpinBox(), QDoubleSpinBox()
        self.ww.setRange(.00001, 1e12)
        self.wl.setRange(-1e12, 1e12)
        for name, spin in [('窗宽', self.ww), ('窗位', self.wl)]:
            window_row.addWidget(QLabel(name))
            window_row.addWidget(spin)
            spin.valueChanged.connect(self.render)
        window_row.addWidget(button('自动调窗', self.auto_window))
        window_row.addWidget(button('全部适应', self.fit_all))
        board_layout.addLayout(window_row)
        grid = QGridLayout()
        self.views, self.spins, self.slice_sliders, self.plane_titles = [], [], [], []
        for axis, name in enumerate(('XY · 原始切片', 'XZ · 固定行', 'YZ · 固定列')):
            panel = QGroupBox(name)
            panel_layout = QVBoxLayout(panel)
            spin = QSpinBox()
            spin.valueChanged.connect(lambda value, a=axis: self.set_position(a, value))
            panel_layout.addWidget(spin)
            view = LabelView(self, axis)
            view.setMinimumSize(250, 220)
            panel_layout.addWidget(view, 1)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 0)
            slider.setToolTip('切换当前幕布的切片；滚轮也可切换')
            slider.valueChanged.connect(lambda value, a=axis: self.slice_changed(a, value))
            panel_layout.addWidget(slider)
            self.views.append(view)
            self.spins.append(spin)
            self.slice_sliders.append(slider)
            self.plane_titles.append(panel)
            grid.addWidget(panel, axis//2, axis % 2)
        overview = QGroupBox('标注状态')
        overview_layout = QVBoxLayout(overview)
        self.summary = QLabel('同一标签体可在三个正交切面编辑。\n\n幕布下方滑块 / 滚轮切层\nCtrl+滚轮缩放 · 中键拖动平移\n多边形：点首点 / 双击 / 右键 / Enter 闭合\n定位工具下右键拖动调窗\n画笔下右键仅擦除当前标签\nCtrl+Z 撤销 · Ctrl+Y 重做\n1 定位 / 2 画笔 / 3 橡皮 / 4 多边形')
        self.summary.setWordWrap(True)
        overview_layout.addWidget(self.summary)
        self.dirty = QLabel('')
        self.dirty.setWordWrap(True)
        overview_layout.addWidget(self.dirty)
        self.cursor_info = QLabel('')
        self.cursor_info.setWordWrap(True)
        overview_layout.addWidget(self.cursor_info)
        grid.addWidget(overview, 1, 1)
        board_layout.addLayout(grid, 1)
        split.addWidget(board)
        split.setStretchFactor(1, 1)
        split.setSizes([345, 1240])
        for key, callback in [('Ctrl+Z', lambda: self.history(False)),
                              ('Ctrl+Y', lambda: self.history(True)),
                              ('Return', self.accept_polygon), ('Enter', self.accept_polygon),
                              ('Ctrl+V', self.paste_polygon),
                              ('Backspace', self.delete_vertex)]:
            shortcut = QShortcut(QKeySequence(key), self, activated=callback)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        for index in range(4):
            shortcut = QShortcut(QKeySequence(str(index+1)), self,
                                 activated=lambda i=index: self.set_tool(('navigate', 'brush', 'erase', 'polygon')[i]))
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.refresh_sources()
        self.update_actions()

    @property
    def busy(self):
        return self.job is not None

    def set_tool(self, key):
        if key != self.tool_key:
            self.cancel_gestures()
            self.tool_key = key
        self.tool_buttons[key].setChecked(True)
        for view in self.views:
            view.viewport().update()

    def set_policy(self, key):
        self.paint_policy = key
        self.render()

    def sync_slice_controls(self):
        if self.session is None:
            return
        for axis, size in enumerate(self.session.labels.shape):
            ids = self.session.source.sequence.indices
            with QSignalBlocker(self.spins[axis]), QSignalBlocker(self.slice_sliders[axis]):
                self.spins[axis].setRange(min(ids), max(ids)) if axis == 0 else self.spins[axis].setRange(0, size-1)
                self.spins[axis].setValue(ids[self.position[0]] if axis == 0 else self.position[axis])
                self.slice_sliders[axis].setRange(0, size-1)
                self.slice_sliders[axis].setValue(self.position[axis])

    def slice_changed(self, axis, position):
        if self.session is None or self.busy:
            return
        self.set_position(axis, self.session.source.sequence.indices[position] if axis == 0 else position)

    def refresh_sources(self):
        available = {key: value for key, value in self.owner.datasets.items()
                     if value.domain == 'reconstruction' and len(value.shape) == 2}
        for key in list(self.sessions):
            if key not in self.owner.datasets:
                self.sessions.pop(key)
        if self.job is not None and self.job.source_id not in available:
            self.cancel_task()
        if self.source_id is not None and self.source_id not in available:
            self.cancel_gestures()
            self.cancel_task()
            self.session = self.source_id = None
            self.label_table.setRowCount(0)
            self.dirty.clear()
            self.render()
        selected = self.source.currentData()
        with QSignalBlocker(self.source):
            self.source.clear()
            self.source.addItem('选择图像管理中的重建数据', None)
            for key, value in available.items():
                self.source.addItem(value.name, key)
                self.source.setItemData(self.source.count()-1, str(value.paths[0]), Qt.ItemDataRole.ToolTipRole)
            self.source.setCurrentIndex(max(0, self.source.findData(selected)))
        self.update_actions()

    def receive_drop(self, mime):
        if mime.hasFormat(MIME_DATASET):
            key = bytes(mime.data(MIME_DATASET)).decode()
            self.refresh_sources()
            row = self.source.findData(key)
            if row > 0:
                self.source.setCurrentIndex(row)
                self.open_source()
            else:
                self.status.setText('请选择重建域单通道数据')
        else:
            self.owner.import_paths(local_paths(mime), -3)

    def open_source(self):
        if self.busy:
            return
        key = self.source.currentData()
        dataset = self.owner.datasets.get(key)
        if dataset is None:
            self.status.setText('请先从下拉框选择重建数据')
            return
        self.cancel_gestures()
        if key in self.sessions:
            self.install_session(key, self.sessions[key])
        else:
            source = ProcessingSource.from_dataset(dataset)
            self.start_task('load', key, lambda cancelled, progress:
                            LabelSession.load(source, cancelled, progress))

    def install_session(self, key, session):
        if self.source_id != key:
            self.last_polygons.clear()
        self.sessions[key] = self.session = session
        self.source_id = key
        self.position = [size//2 for size in session.labels.shape]
        self.source_info.setText(f'{session.source.name}\n{session.labels.shape[2]} × '
            f'{session.labels.shape[1]} × {session.labels.shape[0]} · 标签独立保存')
        self.refresh_labels()
        self.sync_slice_controls()
        self.projection_ready = False
        try:
            geometry = ProjectionGeometry.from_dataset(self.owner.datasets[key])
            expected = tuple(range(geometry.slice_begin, geometry.slice_end+1))
            if tuple(sorted(session.source.sequence.indices)) != expected:
                raise ValueError('缺少部分真实切片，需补齐后生成投影')
            self.projection_ready = True
            spacing = [np.linalg.norm(vector) for vector in
                       (geometry.column_step, geometry.row_step, geometry.slice_step)]
            self.geometry_info.setText(f'几何已就绪 · {len(geometry.matrices)} 个角度\n'
                f'体素间距 XYZ：{spacing[0]:g} / {spacing[1]:g} / {spacing[2]:g} mm')
        except ValueError as error:
            self.geometry_info.setText(f'可勾画和保存；生成投影需补全几何：{error}')
        self.auto_window()
        self.fit_all()
        self.changed()

    def refresh_labels(self):
        active = self.active_label() if self.label_table.rowCount() else 1
        with QSignalBlocker(self.label_table):
            self.label_table.setRowCount(len(self.session.definitions))
            for row, label in enumerate(self.session.definitions):
                value = QTableWidgetItem(str(label.value))
                value.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                self.label_table.setItem(row, 0, value)
                name = QTableWidgetItem(label.name)
                name.setForeground(QColor(*label.color))
                self.label_table.setItem(row, 1, name)
                for column, checked in enumerate((label.visible, label.locked, label.selected), 2):
                    item = QTableWidgetItem()
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                    self.label_table.setItem(row, column, item)
            row = next((i for i, label in enumerate(self.session.definitions) if label.value == active), 0)
            self.label_table.selectRow(row)
        height = (self.label_table.horizontalHeader().height() +
                  self.label_table.rowHeight(0)*min(3, len(self.session.definitions))+4)
        self.label_table.setFixedHeight(height)

    def active_label(self):
        row = self.label_table.currentRow()
        item = self.label_table.item(row, 0) if row >= 0 else None
        return int(item.text()) if item is not None else 1

    def edit_label(self, item):
        if self.session is None or self.busy:
            return
        label = self.session.definitions[item.row()]
        if item.column() == 1:
            label.name = item.text().strip() or f'标签 {label.value}'
        elif item.column() >= 2:
            setattr(label, ('visible', 'locked', 'selected')[item.column()-2],
                    item.checkState() == Qt.CheckState.Checked)
        self.session.revision += 1
        self.changed()

    def pick_color(self, row, column):
        if column != 0 or self.session is None or self.busy:
            return
        label = self.session.definitions[row]
        color = QColorDialog.getColor(QColor(*label.color), self, '标签颜色')
        if color.isValid():
            label.color = color.getRgb()[:3]
            self.session.revision += 1
            self.refresh_labels()
            self.changed()

    def add_label(self):
        if self.session is None or self.busy:
            return
        value = max(label.value for label in self.session.definitions)+1
        if value > 65535:
            return
        colors = [(65, 220, 210), (255, 211, 80), (193, 117, 255), (85, 165, 255)]
        self.session.definitions.append(Label(value, f'标签 {value}', colors[(value-2) % len(colors)], selected=False))
        self.session.revision += 1
        self.refresh_labels()
        self.label_table.selectRow(self.label_table.rowCount()-1)
        self.changed()

    def set_position(self, axis, value):
        if self.session is None or self.busy:
            return
        self.cancel_gestures()
        if axis == 0:
            ids = self.session.source.sequence.indices
            self.position[0] = min(range(len(ids)), key=lambda i: abs(ids[i]-value))
            with QSignalBlocker(self.spins[0]):
                self.spins[0].setValue(ids[self.position[0]])
        else:
            self.position[axis] = value
        self.active_axis = axis
        self.sync_slice_controls()
        self.render()

    def step(self, axis, delta):
        if self.session is None or self.busy:
            return
        target = int(np.clip(self.position[axis]+delta, 0, self.session.labels.shape[axis]-1))
        self.spins[axis].setValue(self.session.source.sequence.indices[target] if axis == 0 else target)

    def navigate(self, axis, point):
        x, y = map(int, point)
        if axis == 0:
            self.position[1:] = [y, x]
        elif axis == 1:
            self.position[0], self.position[2] = y, x
        else:
            self.position[0], self.position[1] = y, x
        self.sync_slice_controls()
        self.render()

    def render(self, *_):
        if not hasattr(self, 'views'):
            return
        if self.session is None:
            for view in self.views:
                view.show_pixels(None, reset=True)
            return
        if self.busy:
            return
        session = self.session
        n, y, x = self.position
        for axis, view in enumerate(self.views):
            raw = plane(session.volume, axis, self.position[axis])
            labels = plane(session.labels, axis, self.position[axis])
            gray = display_pixels(raw, self.ww.value(), self.wl.value(), False)
            rgb = np.repeat(gray[..., None], 3, axis=2)
            alpha = self.opacity.value()/100
            if self.show_labels.isChecked():
                for label in session.definitions:
                    if label.visible:
                        selected = labels == label.value
                        rgb[selected] = np.rint(rgb[selected]*(1-alpha)+np.array(label.color)*alpha).astype(np.uint8)
            if self.threshold_preview.isChecked():
                from .processing import binary_threshold
                selected = binary_threshold(raw, self.threshold.value(), self.comparison.currentData())
                selected &= session.editable(labels, self.active_label(), self.paint_policy)
                rgb[selected] = np.rint(rgb[selected]*.5+np.array([255, 220, 70])*.5).astype(np.uint8)
            view.show_pixels(rgb)
            point = ((x+.5, y+.5) if axis == 0 else (x+.5, n+.5) if axis == 1 else (y+.5, n+.5))
            view.set_navigation_point((point[0]/rgb.shape[1], point[1]/rgb.shape[0]))
        self.cursor_info.setText(f'N={session.source.sequence.indices[n]} · Y={y} · X={x}\n'
            f'原始值：{session.volume[n, y, x]:g} · 标签：{session.labels[n, y, x]}\n'
            '坐标按数组 N/Y/X 显示；不假定患者解剖朝向。')

    def auto_window(self):
        if self.session is None or self.busy:
            return
        width, level = auto_window(self.session.volume[self.position[0]])
        with QSignalBlocker(self.ww), QSignalBlocker(self.wl):
            self.ww.setValue(width)
            self.wl.setValue(level)
        self.render()

    def window_dragged(self, dx, dy):
        self.ww.setValue(max(.00001, self.ww.value()*(1+dx*.01)))
        self.wl.setValue(self.wl.value()+dy*self.ww.value()*.005)

    def fit_all(self):
        for view in self.views:
            view.fit_image()

    def cancel_gestures(self, *_):
        if not hasattr(self, 'views'):
            return
        for view in self.views:
            view.points.clear()
            view.vertex = None
            view.vertex_press = None
            if view.stroke is not None and self.session is not None:
                index, before, *_ = view.stroke
                plane(self.session.labels, view.axis, index)[:] = before
            view.stroke = None
            view.viewport().update()
        self.render()

    def delete_vertex(self):
        view = self.views[self.active_axis]
        if view.points:
            view.points.pop()
            view.viewport().update()

    def paste_polygon(self):
        if self.session is None or self.busy:
            return
        view = self.views[self.active_axis]
        view.points = list(self.last_polygons.get(self.active_axis, []))
        self.set_tool('polygon')
        # Changing the tool cancels any unfinished contour; restore the copy.
        view.points = list(self.last_polygons.get(self.active_axis, []))
        view.viewport().update()

    def accept_polygon(self, *, axis=None):
        if self.session is None or self.busy:
            return
        if axis is not None:
            self.active_axis = axis
        view = self.views[self.active_axis]
        if len(view.points) < 3:
            self.status.setText('轮廓至少需要 3 个顶点')
            return
        index = self.position[view.axis]
        target = plane(self.session.labels, view.axis, index)
        before = target.copy()
        selected = polygon_selection(target.shape, view.points)
        self.session.paint(view.axis, index, selected, self.active_label(), self.paint_policy)
        self.session.commit_plane(view.axis, index, before, '多边形')
        self.last_polygons[view.axis] = list(view.points)
        view.points.clear()
        self.changed()

    def history(self, redo):
        if self.session is None or self.busy:
            return
        self.cancel_gestures()
        self.session.redo() if redo else self.session.undo()
        self.changed()

    def changed(self):
        if self.session is not None:
            self.dirty.setText(('● 标签有未保存更改' if self.session.revision != self.session.saved_revision else
                               '标签文件已保存 / 尚无修改') + f'\n编辑版本 {self.session.revision}')
        self.update_actions()
        self.render()

    def update_actions(self):
        if not hasattr(self, 'cancel_button'):
            return
        self.source.setEnabled(not self.busy)
        for control in (*self.spins, *self.slice_sliders):
            control.setEnabled(self.session is not None and not self.busy)
        self.open_button.setEnabled(not self.busy)
        self.editor.setEnabled(self.session is not None and not self.busy)
        self.output_panel.setEnabled(self.session is not None and not self.busy)
        self.project_button.setEnabled(self.projection_ready and self.session is not None and not self.busy)
        self.cancel_button.setVisible(self.busy)
        if self.session is not None:
            self.undo_button.setEnabled(bool(self.session.undo_stack))
            self.redo_button.setEnabled(bool(self.session.redo_stack))

    def apply_threshold(self, whole):
        if self.session is None or self.busy:
            return
        self.cancel_gestures()
        session, active, policy = self.session, self.active_label(), self.paint_policy
        threshold, comparison = self.threshold.value(), self.comparison.currentData()
        axis, index = (0, None) if whole else (self.active_axis, self.position[self.active_axis])
        self.start_task('edit', self.source_id, lambda cancelled, progress:
                        session.threshold(threshold, comparison, active, policy, cancelled, progress, axis, index))

    def output_values(self):
        if self.output_scope.currentData() == 'active':
            return [self.active_label()]
        return [label.value for label in self.session.definitions if label.selected]

    def export_labels(self):
        if self.session is None or self.busy:
            return
        self.cancel_gestures()
        session, values = self.session, self.output_values()
        self.start_task('export', self.source_id,
            lambda cancelled, progress: session.export_mask(values, cancelled, progress))

    def project_labels(self):
        if self.session is None or self.busy:
            return
        self.cancel_gestures()
        try:
            geometry = ProjectionGeometry.from_dataset(self.owner.datasets[self.source_id])
        except ValueError as error:
            self.status.setText(f'无法生成投影：{error}')
            return
        session, values = self.session, self.output_values()
        def work(cancelled, progress):
            snapshot = session.export_mask(values, cancelled, progress)
            try:
                return project_mask(ProcessingSource.from_dataset(snapshot), geometry, cancelled, progress)
            finally:
                snapshot.close()
        self.start_task('export', self.source_id, work)

    def save_labels(self):
        if self.session is None or self.busy:
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存标签文件', '', '标签文件 (*.npz)')
        if path:
            self.cancel_gestures()
            session = self.session
            self.start_task('save', self.source_id,
                lambda cancelled, progress: session.save(path if path.lower().endswith('.npz') else path+'.npz', cancelled))

    def load_labels(self):
        if self.session is None or self.busy:
            return
        path, _ = QFileDialog.getOpenFileName(self, '载入当前重建的标签文件', '', '标签文件 (*.npz)')
        if path:
            self.cancel_gestures()
            session = self.session
            self.start_task('restore', self.source_id,
                lambda cancelled, progress: session.restore(path, cancelled))

    def start_task(self, kind, source_id, work):
        job = LabelTask(source_id, work)
        stamp = self.epoch
        self.job = job
        self.owner.jobs.append(job)
        job.signals.progress.connect(lambda message: self.status.setText(message)
            if self.job is job and self.epoch == stamp else None)
        job.signals.finished.connect(lambda result, error:
            self.task_finished(job, stamp, kind, result, error))
        self.status.setText('正在处理…')
        self.update_actions()
        QThreadPool.globalInstance().start(job)

    def task_finished(self, job, stamp, kind, result, error):
        if job in self.owner.jobs:
            self.owner.jobs.remove(job)
        if self.job is job:
            self.job = None
        stale = (stamp != self.epoch or job.cancelled.is_set() or job.source_id not in self.owner.datasets)
        if stale:
            if result is not None and hasattr(result, 'close'):
                result.close()
            if stamp == self.epoch:
                self.status.setText('任务已取消；已应用的标签保留，未完成的阈值操作已回滚')
        elif error:
            self.status.setText(f'操作失败：{error}')
        elif kind == 'load' and result is not None:
            self.install_session(job.source_id, result)
            self.status.setText('重建已打开，可在三个切面勾画标签')
        elif kind == 'export' and result is not None:
            self.datasetGenerated.emit(result)
            self.status.setText(f'已加入图像管理：{result.name}；标签继续保留，可修改后再次生成')
        else:
            if kind == 'restore' and self.session is not None:
                self.refresh_labels()
            if kind == 'edit':
                with QSignalBlocker(self.threshold_preview):
                    self.threshold_preview.setChecked(False)
            self.status.setText('标签文件已保存' if kind == 'save' else '操作完成，可撤销继续编辑')
        self.changed()

    def cancel_task(self):
        if self.job is not None:
            self.job.cancelled.set()
            self.status.setText('正在取消…')

    def clear(self):
        self.cancel_gestures()
        self.cancel_task()
        self.epoch += 1
        self.job = None
        self.session = self.source_id = None
        self.sessions.clear()
        self.last_polygons.clear()
        self.projection_ready = False
        self.label_table.setRowCount(0)
        self.source_info.setText('先在数据审查中导入重建数据')
        self.dirty.clear()
        self.cursor_info.clear()
        self.geometry_info.clear()
        for control in (*self.spins, *self.slice_sliders):
            with QSignalBlocker(control):
                control.setRange(0, 0)
        self.status.setText('标注工作台已清空')
        self.render()
        self.update_actions()
