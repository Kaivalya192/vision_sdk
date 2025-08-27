# ================================
# FILE: dexsdk/ui/video_label.py
# ================================
from PyQt5 import QtCore, QtGui, QtWidgets

class VideoLabel(QtWidgets.QLabel):
    roiSelected = QtCore.pyqtSignal(QtCore.QRect)     # existing: rectangular ROI
    polygonSelected = QtCore.pyqtSignal(object)       # NEW: list[QtCore.QPoint] in label coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScaledContents(False)
        self.setMouseTracking(True)

        # keep-aspect rendering rect of the scaled pixmap
        self._last_draw_rect = QtCore.QRect()

        # RECT mode
        self._select_rect_mode = False
        self._origin = QtCore.QPoint()
        self._rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)

        # POLY mode
        self._select_poly_mode = False
        self._poly_pts = []           # list[QPoint]
        self._poly_active = False

    # ---------- public API ----------
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

        # draw onto black canvas so we can overlay guides
        canvas = QtGui.QPixmap(label_size)
        canvas.fill(QtCore.Qt.black)
        painter = QtGui.QPainter(canvas)
        painter.drawPixmap(self._last_draw_rect, pm_scaled)

        # draw polygon preview if in poly mode
        if self._select_poly_mode and self._poly_pts:
            pen = QtGui.QPen(QtCore.Qt.yellow)
            pen.setWidth(2)
            painter.setPen(pen)
            # draw segments
            for i in range(1, len(self._poly_pts)):
                painter.drawLine(self._poly_pts[i-1], self._poly_pts[i])
            # draw vertex markers
            for p in self._poly_pts:
                painter.drawEllipse(p, 3, 3)
        painter.end()
        super().setPixmap(canvas)

    def enable_selection(self, enabled: bool):
        """Back-compat: enable rectangular selection."""
        self.enable_rect_selection(enabled)

    def enable_rect_selection(self, enabled: bool):
        self._select_rect_mode = enabled
        if enabled:
            self._select_poly_mode = False
            self._poly_pts.clear()
            self._poly_active = False
        else:
            self._rubber.hide()

    def enable_polygon_selection(self, enabled: bool):
        """Enable polygon ROI mode (click to add points, right-click or double-click to finish)."""
        self._select_poly_mode = enabled
        self._poly_pts.clear()
        self._poly_active = enabled
        if enabled:
            self._select_rect_mode = False
            self._rubber.hide()
        self.update()

    # ---------- events ----------
    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if self._select_rect_mode and ev.button() == QtCore.Qt.LeftButton:
            self._origin = ev.pos()
            self._rubber.setGeometry(QtCore.QRect(self._origin, QtCore.QSize()))
            self._rubber.show()

        elif self._select_poly_mode:
            # accept points only inside drawn image area
            if ev.button() == QtCore.Qt.LeftButton and self._last_draw_rect.contains(ev.pos()):
                self._poly_pts.append(ev.pos())
                self.update()
            elif ev.button() == QtCore.Qt.RightButton:
                # finish polygon
                self._finish_polygon()

        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        if self._select_rect_mode and not self._origin.isNull():
            self._rubber.setGeometry(QtCore.QRect(self._origin, ev.pos()).normalized())
        self.update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if self._select_rect_mode and ev.button() == QtCore.Qt.LeftButton:
            rect = self._rubber.geometry()
            self._rubber.hide()
            self._select_rect_mode = False
            if rect.width() > 5 and rect.height() > 5:
                self.roiSelected.emit(rect)
        super().mouseReleaseEvent(ev)

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent):
        if self._select_poly_mode:
            self._finish_polygon()
        super().mouseDoubleClickEvent(ev)

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        # ESC cancels polygon capture
        if self._select_poly_mode and ev.key() == QtCore.Qt.Key_Escape:
            self.enable_polygon_selection(False)
        super().keyPressEvent(ev)

    # ---------- helpers ----------
    def _finish_polygon(self):
        if not self._select_poly_mode:
            return
        self._select_poly_mode = False
        self._poly_active = False
        # need at least a triangle
        if len(self._poly_pts) >= 3:
            # emit a plain python list of QPoint (easier to consume)
            self.polygonSelected.emit(list(self._poly_pts))
        self._poly_pts.clear()
        self.update()
