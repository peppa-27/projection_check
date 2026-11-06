"""Independent projection-to-volume workspace and interactive diagnostics."""
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, Qt, QThreadPool, QTimer, Signal, QRectF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QSplitter, QFormLayout, QScrollArea,
    QFrame, QFileDialog, QCheckBox)

from .app import ImageView, button
from .geometry import ProjectionGeometry, GeometryError
from .processing import ProcessingSource, ProcessingCancelled, check_cancelled
from .analysis_core import (build_support, reproject, roi_metrics, difference_counts,
    difference_pixels, ray_profile, export_support, export_envelope, export_reprojection)


class TaskSignals(QObject):
    progress = Signal(str)
    partial = Signal(object)
    finished = Signal(object, object)


class AnalysisTask(QRunnable):
    def __init__(self, kind, stamp, work):
        super().__init__()
        self.kind, self.stamp, self.work = kind, stamp, work
        self.cancelled = Event()
        self.priority = 0
        self.signals = TaskSignals()

    def run(self):
        result, error = None, None
        try:
            result = self.work(self)
        except ProcessingCancelled:
            pass
        except Exception as caught:
            error = str(caught)
        finally:
            self.work = None
        self.signals.finished.emit(result, error)


class AnalysisPlot(QWidget):
    selected = Signal(int)

    def __init__(self, title, mode='line'):
        super().__init__()
        self.title, self.mode = title, mode
        self.x = np.array([])
        self.lines = []
        self.states = None
        self.current = None
        self.threshold = None
        self.setMinimumHeight(110 if mode != 'votes' else 85)
        self.setMouseTracking(True)

    def set_data(self, x=(), lines=(), states=None, current=None, threshold=None):
        self.x = np.asarray(x)
        self.lines = [(np.asarray(y), color) for y, color in lines]
        self.states, self.current, self.threshold = states, current, threshold
        self.update()

    def plot_rect(self):
        return QRectF(36, 29, max(1, self.width()-48), max(1, self.height()-51))

    def px(self, index):
        rect = self.plot_rect()
        if len(self.x) < 2 or self.x[-1] == self.x[0]:
            return rect.center().x()
        return rect.left()+(self.x[index]-self.x[0])/(self.x[-1]-self.x[0])*rect.width()

    def nearest(self, position):
        if not len(self.x):
            return None
        rect = self.plot_rect()
        x = self.x[0] + np.clip((position.x()-rect.left())/rect.width(), 0, 1)*(self.x[-1]-self.x[0])
        return int(np.argmin(abs(self.x-x)))

    def mousePressEvent(self, event):
        index = self.nearest(event.position())
        if event.button() == Qt.MouseButton.LeftButton and index is not None:
            self.selected.emit(index)

    def mouseMoveEvent(self, event):
        index = self.nearest(event.position())
        if index is not None:
            values = [f'{y[index]*100:.2f}%' if np.isfinite(y[index]) else '无有效统计' for y, _ in self.lines]
            if self.states is not None:
                values = [{-1: '视野外/无效', 0: '背景票', 1: '前景票'}[int(self.states[index])]]
            self.setToolTip(f'{self.x[index]:g} · ' + ' / '.join(values))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QColor('#0d1621'))
        p.setPen(QColor('#2a4055'))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(.5, .5, -.5, -.5), 5, 5)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QColor('#c9e5e6'))
        p.drawText(10, 19, self.title)
        rect = self.plot_rect()
        if not len(self.x):
            p.setPen(QColor('#8192a9'))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, '计算后显示；点击曲线可定位')
            return
        p.setPen(QPen(QColor('#2b3b4e'), 1))
        for t in (0., .5, 1.):
            y = rect.bottom()-t*rect.height()
            p.drawLine(rect.left(), y, rect.right(), y)
        p.setPen(QColor('#8192a9'))
        p.drawText(2, int(rect.top()+5), '100%')
        p.drawText(10, int(rect.bottom()), '0%')
        p.drawText(int(rect.left()), self.height()-5, f'{self.x[0]:g}')
        p.drawText(int(rect.right()-40), self.height()-5, f'{self.x[-1]:g}')
        if self.states is not None:
            width = max(1., rect.width()/len(self.x))
            for i, state in enumerate(self.states):
                color = {-1: '#405166', 0: '#ffa645', 1: '#63d3c7'}[int(state)]
                p.fillRect(QRectF(self.px(i)-width/2, rect.top(), width, rect.height()), QColor(color))
        for values, color in self.lines:
            path, started = QPainterPath(), False
            for i, value in enumerate(values):
                if not np.isfinite(value):
                    started = False
                    continue
                if self.mode == 'views' and i > 0 and self.x[i]-self.x[i-1] > 1:
                    started = False
                x, y = self.px(i), rect.bottom()-np.clip(value, 0, 1)*rect.height()
                if started:
                    path.lineTo(x, y)
                else:
                    path.moveTo(x, y)
                    started = True
            p.setPen(QPen(QColor(color), 1.7))
            p.drawPath(path)
        if self.threshold is not None:
            p.setPen(QPen(QColor('#d6e4ec'), 1, Qt.PenStyle.DashLine))
            y = rect.bottom()-self.threshold*rect.height()
            p.drawLine(rect.left(), y, rect.right(), y)
        if self.current is not None and 0 <= self.current < len(self.x):
            p.setPen(QPen(QColor('#ff626c'), 1))
            x = self.px(self.current)
            p.drawLine(x, rect.top(), x, rect.bottom())


def image_panel(title):
    frame = QFrame()
    frame.setObjectName('pane')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(8, 7, 8, 7)
    label = QLabel(title)
    label.setObjectName('paneTag')
    label.setMinimumWidth(0)
    layout.addWidget(label)
    view = ImageView()
    view.setAcceptDrops(False)
    view.viewport().setAcceptDrops(False)
    view.setMinimumSize(90, 90)
    view.hint = '选择投影掩膜并计算支持率'
    layout.addWidget(view, 1)
    return frame, view, label


class ProjectionReconstructionWidget(QWidget):
    datasetGenerated = Signal(object)

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.jobs = []
        self.result = self.series = None
        self.generation = self.projection_version = self.metrics_version = 0
        self.pending_projection = False
        self.source_id = None
        self.manual_geometry = None
        self.roi = None
        self.metrics = None
        self.selected_xyz = self.selected_votes = None
        self.ray = None
        self.profile = None
        self._sync = False
        self.volume_maximized = False
        self._charts_splitter_state = None
        self._projection_timer = QTimer(self)
        self._projection_timer.setSingleShot(True)
        self._projection_timer.setInterval(160)
        self._projection_timer.timeout.connect(self.start_reprojection)
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setSingleShot(True)
        self._metrics_timer.setInterval(100)
        self._metrics_timer.timeout.connect(self.start_metrics)
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        root.addWidget(split)
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(260)
        controls_scroll.setMaximumWidth(340)
        controls = QWidget()
        form = QFormLayout(controls)
        form.setContentsMargins(0, 6, 10, 0)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        title = QLabel('投影域重建 · 一致性分析')
        title.setObjectName('sidebarTitle')
        form.addRow(title)
        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(0)
        self.source_combo.setToolTip('直接选择图像管理中的投影域二值掩膜；也可用下方按钮添加文件或文件夹')
        self.source_combo.currentIndexChanged.connect(self.source_changed)
        form.addRow('输入掩膜', self.source_combo)
        row = QHBoxLayout()
        row.addWidget(button('添加文件', self.open_files))
        row.addWidget(button('添加文件夹', self.open_folder))
        form.addRow(row)
        self.input_dtype = QComboBox()
        self.input_dtype.addItem('沿用数据审查的 Raw 参数', None)
        for label, dtype in [('float16 · 小端', '<f2'), ('uint16 · 小端', '<u2'),
                             ('uint8', '|u1'), ('float32 · 小端', '<f4'), ('int16 · 小端', '<i2')]:
            self.input_dtype.addItem(label, dtype)
        self.input_dtype.setMinimumWidth(0)
        self.input_dtype.setToolTip('只影响之后添加的 Raw 文件；图像与 NPY 使用自带 dtype，尺寸沿用导入参数/XML')
        form.addRow('添加前 Raw dtype', self.input_dtype)
        self.geometry_label = QLabel('几何：自动读取同样本 PTM / XML / INI')
        self.geometry_label.setWordWrap(True)
        self.geometry_label.setObjectName('muted')
        form.addRow(self.geometry_label)
        form.addRow(button('指定几何样本目录…', self.choose_geometry))
        self.resolution = QComboBox()
        for text, factor in [('×4 预览网格', 4), ('×2 较细网格', 2), ('×1 原生网格', 1)]:
            self.resolution.addItem(text, factor)
        self.resolution.setToolTip('所有输入视角完整参与计数；预览网格较粗，细小结构可能变化。原生网格保留 INI 尺寸。')
        form.addRow('空间分辨率', self.resolution)
        self.build_button = button('计算三维支持率', self.start_support)
        form.addRow(self.build_button)
        self.cancel_button = button('取消后台计算', self.cancel)
        form.addRow(self.cancel_button)
        form.addRow(button('清空本页分析', self.reset_analysis))
        self.status = QLabel('选择投影域二值掩膜开始。支持率反映几何支持程度。')
        self.status.setWordWrap(True)
        self.status.setObjectName('muted')
        form.addRow(self.status)
        self.tau = QDoubleSpinBox()
        self.tau.setRange(0, 1)
        self.tau.setDecimals(3)
        self.tau.setSingleStep(.005)
        self.tau.setValue(.96)
        self.tau.setMinimumWidth(0)
        self.tau.valueChanged.connect(self.parameters_changed)
        form.addRow('一致性阈值 τ', self.tau)
        self.tau_slider = QSlider(Qt.Orientation.Horizontal)
        self.tau_slider.setRange(0, 1000)
        self.tau_slider.setValue(960)
        self.tau_slider.valueChanged.connect(lambda v: self.tau.setValue(v/1000))
        form.addRow(self.tau_slider)
        self.min_views = QSpinBox()
        self.min_views.setRange(1, 65535)
        self.min_views.setValue(1)
        self.min_views.valueChanged.connect(self.parameters_changed)
        form.addRow('最少有效视角', self.min_views)
        self.volume_mode = QComboBox()
        self.volume_mode.addItems(['支持率热图', '二值包络', '支持率 + 包络边界', '有效视角覆盖率'])
        self.volume_mode.currentIndexChanged.connect(self.render_volume)
        form.addRow('体数据显示', self.volume_mode)
        self.orientation = QComboBox()
        self.orientation.addItems(['XY · 固定层 N', 'XZ · 固定行 Y', 'YZ · 固定列 X'])
        self.orientation.currentIndexChanged.connect(self.orientation_changed)
        form.addRow('截面方向', self.orientation)
        self.reference_combo = QComboBox()
        self.reference_combo.setMinimumWidth(0)
        self.reference_combo.currentIndexChanged.connect(self.render_volume)
        self.reference_combo.setToolTip('XY 截面可叠加同样本灰度重建作为观察背景；该图像不参与支持率计算。')
        form.addRow('灰度参考图', self.reference_combo)
        self.roi_only = QCheckBox('曲线仅统计框选区域')
        self.roi_only.setChecked(True)
        self.roi_only.toggled.connect(self.schedule_metrics)
        form.addRow(self.roi_only)
        form.addRow(button('清除框选与定位', self.clear_selection))
        form.addRow(button('刷新当前阈值重投影', self.schedule_reprojection))
        self.summary = QLabel('未生成三维包络')
        self.summary.setWordWrap(True)
        form.addRow(self.summary)
        self.export_score_button = button('支持率体 → 暂存图像管理', lambda: self.export('score'))
        self.export_volume_button = button('包络掩膜 → 暂存图像管理', lambda: self.export('volume'))
        self.export_projection_button = button('重投影掩膜 → 暂存图像管理', lambda: self.export('projection'))
        for b in (self.export_score_button, self.export_volume_button, self.export_projection_button):
            form.addRow(b)
        help_text = QLabel('橙色：输入前景未被包络解释\n紫色：重投影超出输入前景\n浅灰：两者均为前景\n未知支持率以灰色显示\n\n点击体素：查各视角投票\n点击投影：查空间射线与深度曲线\n点击曲线：跳转视角或射线体素\n\n差异表示几何不一致，并非真值误差；包络形状可能不唯一。')
        help_text.setWordWrap(True)
        help_text.setObjectName('muted')
        form.addRow(help_text)
        controls_scroll.setWidget(controls)
        split.addWidget(controls_scroll)
        board = QWidget()
        layout = QVBoxLayout(board)
        layout.setContentsMargins(0, 0, 0, 0)
        self.projection_panel = QWidget()
        projection_layout = QVBoxLayout(self.projection_panel)
        projection_layout.setContentsMargins(0, 0, 0, 0)
        projection_row = QHBoxLayout()
        self.projection_views, self.projection_labels = [], []
        for i, text in enumerate(('输入投影掩膜', '三维包络重投影', '差异：橙未解释 / 紫额外')):
            panel, view, label = image_panel(text)
            projection_row.addWidget(panel)
            self.projection_views.append(view)
            self.projection_labels.append(label)
            view.pointPicked.connect(self.pick_projection)
            view.markersChanged.connect(lambda cross, roi, idx=i: self.sync_markers(idx, cross, roi))
            view.viewChanged.connect(lambda state, idx=i: self.sync_view(idx, state))
            view.selectionStarted.connect(self.clear_point)
            view.stepped.connect(self.step_view)
            view.setAcceptDrops(False)
            view.viewport().setAcceptDrops(False)
        projection_layout.addLayout(projection_row, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel('投影视角'))
        self.view_spin = QSpinBox()
        self.view_spin.setRange(0, 0)
        self.view_spin.valueChanged.connect(self.view_changed)
        row.addWidget(self.view_spin)
        self.view_slider = QSlider(Qt.Orientation.Horizontal)
        self.view_slider.setRange(0, 0)
        self.view_slider.valueChanged.connect(self.select_plot_view)
        row.addWidget(self.view_slider, 1)
        self.view_info = QLabel('尚无输入')
        row.addWidget(self.view_info)
        projection_layout.addLayout(row)
        layout.addWidget(self.projection_panel, 4)
        bottom = QSplitter(Qt.Orientation.Horizontal)
        self.charts_splitter = bottom
        bottom.setChildrenCollapsible(False)
        volume_panel, self.volume_view, self.volume_label = image_panel('三维支持率 · 点击查看各视角支持')
        self.volume_panel = volume_panel
        volume_panel.setMinimumWidth(260)
        volume_layout = volume_panel.layout()
        volume_layout.removeWidget(self.volume_label)
        volume_header = QHBoxLayout()
        volume_header.addWidget(self.volume_label, 1)
        self.volume_maximize_button = button('放大', self.toggle_volume_maximize)
        self.volume_maximize_button.setCheckable(True)
        self.volume_maximize_button.setFixedWidth(48)
        self.volume_maximize_button.setToolTip('占满本页图像与曲线区域；再次点击还原，Esc 也可还原；左侧工具保留')
        volume_header.addWidget(self.volume_maximize_button)
        volume_layout.insertLayout(0, volume_header)
        row = QHBoxLayout()
        self.plane_spin = QSpinBox()
        self.plane_spin.setRange(0, 0)
        self.plane_spin.valueChanged.connect(self.plane_changed)
        row.addWidget(self.plane_spin)
        self.plane_slider = QSlider(Qt.Orientation.Horizontal)
        self.plane_slider.setRange(0, 0)
        self.plane_slider.valueChanged.connect(self.plane_spin.setValue)
        row.addWidget(self.plane_slider, 1)
        volume_layout.addLayout(row)
        self.voxel_info = QLabel('点击体素显示坐标、支持率及前景/背景票数')
        self.voxel_info.setWordWrap(True)
        self.voxel_info.setObjectName('muted')
        self.support_legend = QLabel('深蓝 0% → 青色 80% → 金色 95% → 浅黄 100%；灰色未知')
        self.support_legend.setWordWrap(True)
        self.support_legend.setObjectName('muted')
        volume_layout.addWidget(self.support_legend)
        volume_layout.addWidget(self.voxel_info)
        self.volume_view.pointPicked.connect(self.pick_volume)
        self.volume_view.stepped.connect(lambda v: self.plane_spin.setValue(self.plane_spin.value()+v))
        bottom.addWidget(volume_panel)
        charts = QWidget()
        self.charts_panel = charts
        charts.setMinimumWidth(300)
        charts_layout = QVBoxLayout(charts)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        self.error_plot = AnalysisPlot('逐视角差异 · 橙未解释 / 紫额外', 'views')
        self.vote_plot = AnalysisPlot('体素投票 · 青前景 / 橙背景 / 灰无效', 'votes')
        self.ray_plot = AnalysisPlot('射线支持率 · 横轴为沿射线距离 mm')
        self.error_plot.selected.connect(self.select_plot_view)
        self.vote_plot.selected.connect(self.select_plot_view)
        self.ray_plot.selected.connect(self.select_ray_cell)
        for plot in (self.error_plot, self.vote_plot, self.ray_plot):
            charts_layout.addWidget(plot, 1)
        bottom.addWidget(charts)
        bottom.setStretchFactor(0, 1)
        bottom.setStretchFactor(1, 1)
        bottom.setSizes([650, 600])
        QTimer.singleShot(0, lambda: bottom.setSizes([650, 600]))
        layout.addWidget(bottom, 6)
        split.addWidget(board)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([290, 1300])
        self.refresh_sources()
        self.update_actions()

    def toggle_volume_maximize(self):
        self.set_volume_maximized(not self.volume_maximized)

    def set_volume_maximized(self, maximized):
        if maximized == self.volume_maximized:
            return
        self.volume_maximized = maximized
        if maximized:
            self._charts_splitter_state = self.charts_splitter.saveState()
        self.projection_panel.setVisible(not maximized)
        self.charts_panel.setVisible(not maximized)
        if not maximized and self._charts_splitter_state is not None:
            self.charts_splitter.restoreState(self._charts_splitter_state)
            self._charts_splitter_state = None
        self.volume_maximize_button.setChecked(maximized)
        self.volume_maximize_button.setText('还原' if maximized else '放大')

    def current_source(self):
        return self.host.datasets.get(self.source_combo.currentData())

    def ids(self):
        source = self.current_source()
        return self.result.indices if self.result is not None else (tuple(sorted(source.indices)) if source else ())

    def current_position(self):
        ids = self.ids()
        return ids.index(self.view_spin.value()) if self.view_spin.value() in ids else None

    def refresh_sources(self, preferred=None):
        old, reference = self.source_combo.currentData(), self.reference_combo.currentData()
        masks = [d for d in self.host.datasets.values()
                 if d.domain == 'projection' and d.mask and len(d.shape) == 2]
        references = [d for d in self.host.datasets.values()
                      if d.domain == 'reconstruction' and not d.mask and not d.generated and len(d.shape) == 2]
        with QSignalBlocker(self.source_combo), QSignalBlocker(self.reference_combo):
            self.source_combo.clear()
            self.source_combo.addItem('请选择投影域二值掩膜', None)
            for d in masks:
                self.source_combo.addItem(f'{d.name} · {len(d.indices)} 帧 · {d.dtype}', d.id)
                self.source_combo.setItemData(self.source_combo.count()-1, str(d.paths[0]), Qt.ItemDataRole.ToolTipRole)
            chosen = preferred if any(d.id == preferred for d in masks) else old
            if chosen is None and masks:
                chosen = masks[-1].id
            self.source_combo.setCurrentIndex(max(0, self.source_combo.findData(chosen)))
            self.reference_combo.clear()
            self.reference_combo.addItem('无参考图（只分析掩膜）', None)
            for d in references:
                self.reference_combo.addItem(d.name, d.id)
            self.reference_combo.setCurrentIndex(max(0, self.reference_combo.findData(reference)))
        if self.source_combo.currentData() != self.source_id:
            self.source_changed()
        elif reference != self.reference_combo.currentData() and self.result is not None:
            self.render_volume()
        self.update_actions()

    def import_masks(self, paths):
        self.host.import_paths(paths, target=-2, analysis_dtype=self.input_dtype.currentData())

    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, '添加投影掩膜', '',
            '图像与掩膜 (*.prj *.bin *.raw *.npy *.png *.tif *.tiff *.bmp)')
        if paths:
            self.import_masks(paths)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, '添加投影掩膜文件夹')
        if path:
            self.import_masks([path])

    def choose_geometry(self):
        path = QFileDialog.getExistingDirectory(self, '选择含 proj 与 slice 几何文件的样本目录')
        if path:
            self.clear()
            self.manual_geometry = Path(path)
            self.geometry_label.setText(f'指定几何：{path}')

    def source_changed(self, *_):
        self.clear()
        self.source_id = self.source_combo.currentData()
        self.manual_geometry = None
        source = self.current_source()
        if source:
            with QSignalBlocker(self.view_spin), QSignalBlocker(self.view_slider):
                self.view_spin.setRange(min(source.indices), max(source.indices))
                self.view_spin.setValue(source.indices[0])
                self.view_slider.setRange(0, len(source.indices)-1)
                self.view_slider.setValue(0)
            self.geometry_label.setText('几何：自动读取同样本 PTM / XML / INI')
            self.view_changed()
        self.update_actions()

    def update_actions(self):
        support_busy = any(j.kind == 'support' for j in self.jobs)
        export_busy = any(j.kind == 'export' for j in self.jobs)
        self.build_button.setEnabled(self.current_source() is not None and not support_busy)
        self.cancel_button.setEnabled(bool(self.jobs))
        ready = self.result is not None
        self.export_score_button.setEnabled(ready)
        self.export_volume_button.setEnabled(ready and not export_busy)
        self.export_projection_button.setEnabled(self.series is not None and bool(self.series.ready.all()))

    def stamp(self, kind):
        if kind == 'projection':
            return self.generation, self.projection_version
        if kind == 'metrics':
            return self.generation, self.projection_version, self.metrics_version
        return (self.generation,)

    def launch(self, kind, work):
        job = AnalysisTask(kind, self.stamp(kind), work)
        self.jobs.append(job)
        job.signals.progress.connect(lambda text, j=job: self.task_progress(j, text))
        job.signals.partial.connect(lambda data, j=job: self.task_partial(j, data))
        job.signals.finished.connect(lambda result, error, j=job: self.task_finished(j, result, error))
        QThreadPool.globalInstance().start(job)
        self.update_actions()
        return job

    def task_progress(self, job, text):
        if job.stamp == self.stamp(job.kind) and not job.cancelled.is_set():
            self.status.setText(text)

    def start_support(self):
        source = self.current_source()
        if source is None or any(j.kind == 'support' for j in self.jobs):
            return
        try:
            snapshot = ProcessingSource.from_dataset(source)
            if self.manual_geometry:
                folder = self.manual_geometry
                folder = folder if (folder/'PTM.dat').is_file() else folder/'proj'
                snapshot = replace(snapshot, geometry_paths=(folder/'projection_0.prj',))
            descriptor = SimpleNamespace(paths=snapshot.sequence.paths, geometry_paths=snapshot.geometry_paths)
            geometry = ProjectionGeometry.from_projection_dataset(descriptor)
        except Exception as error:
            self.status.setText(f'无法开始：{error}')
            return
        self.clear()
        factor = self.resolution.currentData()
        self.render_projections()
        self.geometry_label.setText(f'{geometry.root.name}\n{geometry.coordinate_convention}\n'
            f'输入 {len(source.indices)}/{len(geometry.matrices)} 个视角；未导入帧不参与计票')
        self.geometry_label.setToolTip(str(geometry.root))
        self.status.setText('准备计算完整支持率…')
        self.launch('support', lambda job: build_support(snapshot, geometry, factor,
                    job.cancelled, job.signals.progress.emit))

    def task_partial(self, job, data):
        if job.stamp != self.stamp(job.kind) or job.cancelled.is_set():
            return
        kind, payload = data
        if kind == 'series':
            self.series = payload
            self.metrics = np.full((len(payload.ready), 3), -1, np.int64)
            self.update_summary()
        elif kind == 'view' and self.series is not None:
            self.metrics[payload] = difference_counts(self.result.masks[payload], self.series.array[payload], self.active_roi())
            self.update_error_plot()
        self.render_projections()
        self.update_actions()

    def task_finished(self, job, result, error):
        if job in self.jobs:
            self.jobs.remove(job)
        current = job.stamp == self.stamp(job.kind) and not job.cancelled.is_set()
        if not current:
            if getattr(result, 'generated', False):
                result.close()
        elif error:
            self.status.setText(f'计算失败：{error}')
            self.status.setToolTip(error)
        elif result is not None:
            if job.kind == 'support':
                self.result = result
                with QSignalBlocker(self.min_views):
                    self.min_views.setMaximum(len(result.indices))
                    self.min_views.setValue(min(self.min_views.value(), len(result.indices)))
                self.orientation_changed()
                self.status.setText(f'支持率完成 · 网格 {result.shape} · 采样 ×{result.factor}')
                self.schedule_reprojection()
            elif job.kind == 'projection':
                self.series = result
                self.status.setText(f'全部 {len(result.ready)} 个输入视角重投影完成 · τ={result.threshold:.3f}')
                self.schedule_metrics()
                self.update_summary()
            elif job.kind == 'metrics':
                self.metrics = result
                self.update_error_plot()
            elif job.kind == 'export':
                self.datasetGenerated.emit(result)
                self.status.setText('结果已暂存到数据审查的图像管理')
        if job.kind == 'projection' and self.pending_projection and not any(j.kind == 'projection' for j in self.jobs):
            self._projection_timer.start(0)
        self.update_actions()
        # Break signal closures so snapshots and temporary resources are released.
        job.signals.progress.disconnect()
        job.signals.partial.disconnect()
        job.signals.finished.disconnect()

    def parameters_changed(self, *_):
        with QSignalBlocker(self.tau_slider):
            self.tau_slider.setValue(round(self.tau.value()*1000))
        self.render_volume()
        if self.profile is not None:
            self.ray_plot.set_data(self.profile[0], [(self.profile[1], '#63d3c7')], threshold=self.tau.value())
        self.schedule_reprojection()

    def schedule_reprojection(self, *_):
        if self.result is None:
            return
        self.projection_version += 1
        self.metrics_version += 1
        for job in self.jobs:
            if job.kind in ('projection', 'metrics'):
                job.cancelled.set()
        self.series = self.metrics = None
        self.pending_projection = True
        self.error_plot.set_data()
        self.summary.setText(f'当前 τ={self.tau.value():.3f}；等待重投影统计更新')
        self.render_projections()
        self._projection_timer.start()
        self.update_actions()

    def start_reprojection(self):
        if self.result is None or not self.pending_projection:
            return
        if any(j.kind == 'projection' for j in self.jobs):
            return
        self.pending_projection = False
        source, threshold, minimum = self.result, self.tau.value(), self.min_views.value()
        job = self.launch('projection', lambda task: reproject(source, threshold, minimum,
            task.cancelled, task.signals.progress.emit, task.signals.partial.emit, lambda: task.priority))
        job.priority = self.current_position() or 0

    def active_roi(self):
        return self.roi if self.roi_only.isChecked() else None

    def schedule_metrics(self, *_):
        self.metrics_version += 1
        for job in self.jobs:
            if job.kind == 'metrics':
                job.cancelled.set()
        if self.series is not None:
            self.metrics = np.full((len(self.series.ready), 3), -1, np.int64)
            self.error_plot.set_data()
            self._metrics_timer.start()

    def start_metrics(self):
        if self.series is None:
            return
        series, roi = self.series, self.active_roi()
        self.launch('metrics', lambda job: roi_metrics(series, roi, job.cancelled))

    def update_error_plot(self):
        if self.result is None or self.metrics is None:
            return
        c = self.metrics.astype(float)
        known = (c >= 0).all(axis=1)
        missing, extra = np.full(len(c), np.nan), np.full(len(c), np.nan)
        np.divide(c[:, 1], c[:, 0]+c[:, 1], out=missing, where=known & (c[:, 0]+c[:, 1] > 0))
        np.divide(c[:, 2], c[:, 0]+c[:, 2], out=extra, where=known & (c[:, 0]+c[:, 2] > 0))
        self.error_plot.set_data(self.result.indices, [(missing, '#ffa645'), (extra, '#c175ff')],
                                 current=self.current_position())
        self.update_summary()

    def update_summary(self):
        if self.series is None:
            return
        s = self.series
        volume = s.volume_count * s.source.voxel_mm3
        text = f'τ={s.threshold:.3f} · 有效视角≥{s.min_views}\n包络 {s.volume_count:,} 体素 / {volume:.2f} mm³\n'
        if self.metrics is not None:
            known = (self.metrics >= 0).all(axis=1)
            sums = self.metrics[known].sum(axis=0)
            miss = f'{100*sums[1]/(sums[0]+sums[1]):.2f}%' if sums[0]+sums[1] else '无前景'
            extra = f'{100*sums[2]/(sums[0]+sums[2]):.2f}%' if sums[0]+sums[2] else '无前景'
            text += f'未解释 {miss} · 额外 {extra}\n统计 {known.sum()}/{len(known)} 帧' + (' · 框内' if self.active_roi() else ' · 全图')
        self.summary.setText(text)

    def select_plot_view(self, index):
        ids = self.ids()
        if 0 <= index < len(ids):
            self.view_spin.setValue(ids[index])

    def step_view(self, amount):
        ids = self.ids()
        if ids:
            index = self.current_position() or 0
            self.select_plot_view(max(0, min(len(ids)-1, index+amount)))

    def view_changed(self, *_):
        index = self.current_position()
        if index is not None:
            with QSignalBlocker(self.view_slider):
                self.view_slider.setValue(index)
            for job in self.jobs:
                if job.kind == 'projection':
                    job.priority = index
        self.render_projections()
        self.update_error_plot()
        if getattr(self, 'ray_origin', None) is not None and self.result is not None:
            self.set_ray(self.ray_origin)
        else:
            self.refresh_markers()
            self.refresh_votes()

    def render_projections(self):
        index = self.current_position()
        frame = None
        if index is not None:
            if self.result is not None:
                frame = self.result.masks[index]
            elif self.current_source() is not None:
                try:
                    raw = self.current_source().frame(self.view_spin.value())
                    frame = (raw != 0).astype(np.uint8)
                except Exception:
                    pass
        self.projection_views[0].show_pixels(None if frame is None else frame.astype(np.uint8)*255)
        ready = index is not None and self.series is not None and self.series.ready[index]
        if ready:
            reconstructed = self.series.array[index]
            self.projection_views[1].show_pixels(reconstructed.astype(np.uint8)*255)
            self.projection_views[2].show_pixels(difference_pixels(frame, reconstructed))
            counts = difference_counts(frame, reconstructed, self.active_roi())
            self.view_info.setText(f'N={self.view_spin.value()} · 未解释 {counts[1]} / 额外 {counts[2]} 像素')
        else:
            for view in self.projection_views[1:]:
                view.hint = '等待当前阈值重投影…' if self.result is not None else '计算支持率后显示'
                view.show_pixels(None)
            self.view_info.setText(f'N={self.view_spin.value()}' if frame is not None else '输入帧缺失')
        self.refresh_markers()

    def sync_view(self, source, state):
        if self._sync:
            return
        self._sync = True
        try:
            for i, view in enumerate(self.projection_views):
                if i != source:
                    view.apply_view_state(state)
        finally:
            self._sync = False

    def sync_markers(self, source, crosshair, roi):
        self.roi = roi
        for view in self.projection_views:
            view.set_markers(None, roi)
        self.schedule_metrics()
        self.render_projections()

    def orientation_changed(self, *_):
        if self.result is None:
            return
        axis = (2, 1, 0)[self.orientation.currentIndex()]
        count = self.result.shape[::-1][axis]
        begin = self.result.geometry.slice_begin if axis == 2 else 0
        with QSignalBlocker(self.plane_spin), QSignalBlocker(self.plane_slider):
            self.plane_spin.setRange(begin, begin+count-1)
            self.plane_slider.setRange(begin, begin+count-1)
            value = int(self.selected_xyz[axis])+begin if self.selected_xyz is not None else begin+count//2
            self.plane_spin.setValue(value)
            self.plane_slider.setValue(value)
            self.plane_spin.setPrefix(('N=', 'Y=', 'X=')[self.orientation.currentIndex()])
        self.reference_combo.setEnabled(self.orientation.currentIndex() == 0)
        self.render_volume()

    def plane_changed(self, *_):
        with QSignalBlocker(self.plane_slider):
            self.plane_slider.setValue(self.plane_spin.value())
        self.render_volume()

    def plane_index(self):
        return self.plane_spin.value() - (self.result.geometry.slice_begin if self.orientation.currentIndex() == 0 else 0)

    def plane(self, array):
        index, orientation = self.plane_index(), self.orientation.currentIndex()
        return array[index] if orientation == 0 else (array[:, index, :] if orientation == 1 else array[:, :, index])

    def reference_pixels(self):
        if self.orientation.currentIndex() != 0:
            return None
        dataset = self.host.datasets.get(self.reference_combo.currentData())
        if dataset is None:
            return None
        from .data import auto_window, display_pixels
        try:
            reference = ProjectionGeometry.from_dataset(dataset)
            g = self.result.geometry
            if reference.root != g.root:
                raise GeometryError('参考图与输入掩膜不属于同一个样本')
            basis = np.column_stack((reference.column_step, reference.row_step, reference.slice_step))
            start = np.linalg.solve(basis, g.origin + self.plane_index()*g.slice_step-reference.origin)
            dx, dy = np.linalg.solve(basis, g.column_step), np.linalg.solve(basis, g.row_step)
            if max(abs(dx[1:]).max(), abs(dy[[0, 2]]).max()) > 1e-5:
                raise GeometryError('参考图的 XY 方向与分析网格不同')
            n = reference.slice_begin + int(np.floor(start[2]+.5))
            frame = dataset.frame(n)
            y = np.clip(np.floor(start[1]+np.arange(g.slice_shape[0])*dy[1]+.5).astype(int), 0, frame.shape[0]-1)
            x = np.clip(np.floor(start[0]+np.arange(g.slice_shape[1])*dx[0]+.5).astype(int), 0, frame.shape[1]-1)
            sample = frame[y[:, None], x[None, :]]
            pixels = display_pixels(sample, *auto_window(frame))
            return np.repeat(pixels[..., None], 3, axis=2)
        except Exception as error:
            self.reference_combo.setToolTip(f'参考图暂不可用：{error}')
            return None

    def render_volume(self, *_):
        if self.result is None:
            return
        score, hits, valid = self.plane(self.result.score), self.plane(self.result.hits), self.plane(self.result.valid)
        binary = (valid >= self.min_views.value()) & (valid > 0) & (hits.astype(float)+1e-9 >= self.tau.value()*valid)
        mode = self.volume_mode.currentIndex()
        values = valid/len(self.result.indices) if mode == 3 else np.nan_to_num(score)
        stops = np.array([0., .5, .8, .95, 1.])
        colors = np.array([[10, 18, 32], [37, 91, 145], [34, 163, 157], [243, 178, 70], [255, 248, 205]])
        pixels = np.stack([np.interp(values, stops, colors[:, c]) for c in range(3)], axis=-1).astype(np.uint8)
        reference = self.reference_pixels()
        if mode == 1:
            pixels = np.repeat((binary.astype(np.uint8)*255)[..., None], 3, axis=2)
            if reference is not None:
                pixels = reference.copy()
                pixels[binary] = (reference[binary]*.25 + np.array([99, 211, 199])*.75).astype(np.uint8)
        elif reference is not None:
            alpha = np.where(values > 0, .6, 0.)[..., None]
            pixels = (pixels*alpha + reference*(1-alpha)).astype(np.uint8)
        if mode == 2:
            inside = binary.copy()
            inside[1:] &= binary[:-1]
            inside[:-1] &= binary[1:]
            inside[:, 1:] &= binary[:, :-1]
            inside[:, :-1] &= binary[:, 1:]
            pixels[binary & ~inside] = [255, 95, 108]
        pixels[valid == 0] = [65, 76, 91]
        self.volume_view.show_pixels(pixels)
        if mode == 1:
            self.support_legend.setText('白色前景 / 黑色背景；参考图叠加时前景为青色；灰色未知')
        else:
            prefix = '覆盖率=有效视角/输入视角；' if mode == 3 else ''
            self.support_legend.setText(prefix+'深蓝 0% → 青色 80% → 金色 95% → 浅黄 100%；灰色未知')
        g = self.result.geometry
        self.volume_label.setText(f'{self.volume_mode.currentText()} · 采样 ×{self.result.factor} · τ={self.tau.value():.3f}')
        self.volume_label.setToolTip(f'网格 {self.result.shape} · 体素体积 {self.result.voxel_mm3:g} mm³\n{g.coordinate_convention}')
        self.refresh_markers()

    def pick_volume(self, normalized):
        if self.result is None:
            return
        self.ray = self.profile = self.ray_origin = None
        self.ray_plot.set_data()
        horizontal, vertical, fixed = ((0, 1, 2), (0, 2, 1), (1, 2, 0))[self.orientation.currentIndex()]
        counts = self.result.shape[::-1]
        xyz = np.zeros(3, int)
        xyz[fixed] = self.plane_index()
        xyz[horizontal] = min(counts[horizontal]-1, int(normalized[0]*counts[horizontal]))
        xyz[vertical] = min(counts[vertical]-1, int(normalized[1]*counts[vertical]))
        self.selected_xyz = xyz
        self.refresh_votes()
        self.refresh_markers()

    def refresh_votes(self):
        if self.result is None or self.selected_xyz is None:
            self.vote_plot.set_data()
            return
        states, _, _ = self.result.votes(self.selected_xyz)
        self.selected_votes = states
        self.vote_plot.set_data(self.result.indices, states=states, current=self.current_position())
        x, y, z = self.selected_xyz
        h, v = int(self.result.hits[z, y, x]), int(self.result.valid[z, y, x])
        g = self.result.geometry
        world = g.origin + x*g.column_step + y*g.row_step + z*g.slice_step
        support = f'{100*h/v:.2f}%' if v else '未知'
        self.voxel_info.setText(f'网格 [N,Y,X]=[{z+g.slice_begin},{y},{x}] · 支持率 {support}\n'
            f'前景 {h} / 背景 {v-h} / 有效 {v}；输入 {len(states)} 帧\n'
            f'XYZ(mm)=({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f})')
        if self.result.factor > 1:
            native = self.result.native_geometry
            basis = np.column_stack((native.column_step, native.row_step, native.slice_step))
            raw = np.linalg.solve(basis, world-native.origin)
            self.voxel_info.setText(self.voxel_info.text()+
                f'\n原始网格约 [N,Y,X]=[{raw[2]+native.slice_begin:.1f},{raw[1]:.1f},{raw[0]:.1f}]')

    def pick_projection(self, normalized):
        if self.result is None:
            self.status.setText('请先计算支持率，再分析投影射线')
            return
        self.set_ray(normalized)

    def set_ray(self, normalized):
        try:
            self.ray_origin = normalized
            self.ray = self.result.geometry.backproject(normalized, self.view_spin.value(), self.result.masks.shape[1:])
            self.profile = ray_profile(self.result, self.ray)
            self.selected_xyz = None
            self.vote_plot.set_data()
            self.ray_plot.set_data(self.profile[0], [(self.profile[1], '#63d3c7')], threshold=self.tau.value())
            self.voxel_info.setText(f'投影 N={self.view_spin.value()} 的空间射线；点击深度曲线选择经过的体素\n'
                                   '曲线支持率按经过体素的中心统计，不自动猜测唯一深度')
            self.refresh_markers()
        except GeometryError as error:
            self.clear_point()
            self.voxel_info.setText(f'当前射线不可用：{error}')

    def select_ray_cell(self, index):
        if self.profile is None or not 0 <= index < len(self.profile[2]):
            return
        self.selected_xyz = self.profile[2][index]
        axis = (2, 1, 0)[self.orientation.currentIndex()]
        begin = self.result.geometry.slice_begin if axis == 2 else 0
        self.plane_spin.setValue(int(self.selected_xyz[axis])+begin)
        self.refresh_votes()
        self.ray_plot.set_data(self.profile[0], [(self.profile[1], '#63d3c7')], current=index, threshold=self.tau.value())
        self.refresh_markers()

    def refresh_markers(self):
        if not hasattr(self, 'volume_view'):
            return
        for view in self.projection_views:
            view.set_reference_point()
        self.volume_view.set_reference_point()
        self.volume_view.set_ray_trace()
        if self.result is None:
            return
        horizontal, vertical, fixed = ((0, 1, 2), (0, 2, 1), (1, 2, 0))[self.orientation.currentIndex()]
        counts = np.array(self.result.shape[::-1])
        if self.ray is not None:
            for view in self.projection_views:
                view.set_reference_point(self.ray_origin)
            a, b = self.ray.local_endpoints
            def normalized(p):
                return ((p[horizontal]+.5)/counts[horizontal], (p[vertical]+.5)/counts[vertical])
            trace = (normalized(a), normalized(b))
            delta = b-a
            line, point = None, None
            if abs(delta[fixed]) < 1e-10:
                if abs(a[fixed]-self.plane_index()) < 1e-8:
                    line = trace
            else:
                t = (self.plane_index()-a[fixed])/delta[fixed]
                if 0 <= t <= 1:
                    point = normalized(a+t*delta)
            self.volume_view.set_ray_trace(trace, line)
            self.volume_view.set_reference_point(point)
        elif self.selected_xyz is not None:
            index = self.current_position()
            if index is not None:
                _, uv, good = self.result.votes(self.selected_xyz)
                if good[index]:
                    h, w = self.result.masks.shape[1:]
                    point = tuple((uv[index]+.5)/[w, h])
                    for view in self.projection_views:
                        view.set_reference_point(point)
            if self.selected_xyz[fixed] == self.plane_index():
                self.volume_view.set_reference_point(((self.selected_xyz[horizontal]+.5)/counts[horizontal],
                                                     (self.selected_xyz[vertical]+.5)/counts[vertical]))

    def clear_point(self):
        self.ray = self.profile = self.ray_origin = self.selected_xyz = self.selected_votes = None
        self.vote_plot.set_data()
        self.ray_plot.set_data()
        self.voxel_info.setText('点击体素查投票，点击投影查射线')
        for view in [*self.projection_views, self.volume_view]:
            view.set_markers(None, view.selection)
            view.set_reference_point()
            view.set_ray_trace()

    def clear_selection(self):
        self.clear_point()
        self.roi = None
        for view in self.projection_views:
            view.set_markers()
        self.schedule_metrics()

    def export(self, kind):
        if self.result is None:
            return
        if kind == 'score':
            self.datasetGenerated.emit(export_support(self.result))
            self.status.setText('支持率体已暂存到数据审查的图像管理')
        elif kind == 'projection' and self.series is not None:
            try:
                self.datasetGenerated.emit(export_reprojection(self.series))
                self.status.setText('重投影掩膜已暂存到数据审查的图像管理')
            except ValueError as error:
                self.status.setText(str(error))
        elif kind == 'volume' and not any(j.kind == 'export' for j in self.jobs):
            source, threshold, minimum = self.result, self.tau.value(), self.min_views.value()
            self.launch('export', lambda job: export_envelope(source, threshold, minimum, job.cancelled))

    def cancel(self):
        for job in self.jobs:
            job.cancelled.set()
        self.generation += 1
        self.projection_version += 1
        self.pending_projection = False
        self._projection_timer.stop()
        self._metrics_timer.stop()
        self.series = self.metrics = None
        self.error_plot.set_data()
        self.status.setText('后台计算已取消；已完成的支持率保留')
        self.render_projections()
        self.update_actions()

    def clear(self):
        self.set_volume_maximized(False)
        for job in self.jobs:
            job.cancelled.set()
        self.generation += 1
        self.projection_version += 1
        self.metrics_version += 1
        self.pending_projection = False
        self._projection_timer.stop()
        self._metrics_timer.stop()
        self.result = self.series = self.metrics = None
        self.roi = None
        self.clear_point()
        self.error_plot.set_data()
        for view in [*self.projection_views, self.volume_view]:
            view.show_pixels(None, reset=True)
            view.set_markers()
        self.summary.setText('未生成三维包络')
        self.voxel_info.setText('点击体素显示坐标、支持率及前景/背景票数')
        self.status.setText('选择投影掩膜后计算支持率；切换页面会保留分析状态')

    def reset_analysis(self):
        self.clear()
        self.render_projections()
        self.update_actions()

    def fit_all(self):
        for view in [*self.projection_views, self.volume_view]:
            view.fit_image()
