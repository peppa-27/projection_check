"""Startup placement in Qt logical coordinates, including window decorations."""
from PySide6.QtCore import QMargins, QRect, QSize


def startup_geometry(available, margins=QMargins(), preferred=QSize(1680, 960)):
    # Leave a small gap to the taskbar and monitor edges. Never multiply by DPR:
    # QScreen, QWidget and these frame margins already use logical coordinates.
    gap = min(12, max(0, min(available.width(), available.height()) // 40))
    area = available.adjusted(gap, gap, -gap, -gap)
    limit = QSize(max(1, area.width()-margins.left()-margins.right()),
                  max(1, area.height()-margins.top()-margins.bottom()))
    size = preferred.boundedTo(limit)
    frame_width = size.width()+margins.left()+margins.right()
    frame_height = size.height()+margins.top()+margins.bottom()
    rect = QRect(area.left()+(area.width()-frame_width)//2+margins.left(),
                 area.top()+(area.height()-frame_height)//2+margins.top(),
                 size.width(), size.height())
    return rect, QSize(1360, 780).boundedTo(limit)
