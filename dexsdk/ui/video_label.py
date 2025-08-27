
# ===============================
# FILE: dexsdk/ui/video_label.py
# ===============================
from PyQt5 import QtCore, QtGui, QtWidgets

class VideoLabel(QtWidgets.QLabel):
    """QLabel that draws a keep-aspect pixmap and exposes the draw rect.
    It also supports click-drag ROI selection in label coordinates and
    emits `roiSelected(QRect)` on mouse release.
    """
    roiSelected = QtCore.pyqtSignal(QtCore.QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setMouseTracking(True)
        self._select_mode = False
        self._origin = QtCore.QPoint()
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._last_draw_rect = QtCore.QRect()

    def setPixmapKeepAspect(self, pm: QtGui.QPixmap):
        if pm.isNull():
            super().setPixmap(pm)
            self._last_draw_rect = QtCore.QRect()
            return
        label_size = self.size()
        pm_scaled = pm.scaled(label_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        x = (label_size.width() - pm_scaled.width()) // 2
        y = (label_size.height() - pm_scaled.height()) // 2
        self._last_draw_rect = QtCore.QRect(x, y, pm_scaled.width(), pm_scaled.height())
        canvas = QtGui.QPixmap(label_size)
        canvas.fill(QtCore.Qt.black)
        painter = QtGui.QPainter(canvas)
        painter.drawPixmap(self._last_draw_rect, pm_scaled)
        painter.end()
        super().setPixmap(canvas)

    def enable_selection(self, enabled: bool):
        self._select_mode = enabled
        if not enabled:
            self._rubber.hide()

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self._select_mode and ev.button() == QtCore.Qt.LeftButton:
            self._origin = ev.pos()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._select_mode and not self._origin.isNull():
            self._rubber.setGeometry(QtCore.QRect(self._origin, ev.pos()).normalized())
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self._select_mode and ev.button() == QtCore.Qt.LeftButton:
            rect = self._rubber.geometry()
            self._rubber.hide()
            self._select_mode = False
            if rect.width() > 5 and rect.height() > 5:
                self.roiSelected.emit(rect)
        super().mouseReleaseEvent(ev)

