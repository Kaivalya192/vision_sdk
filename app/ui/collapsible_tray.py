from PyQt5 import QtCore, QtWidgets


class CollapsibleTray(QtWidgets.QWidget):
    toggled = QtCore.pyqtSignal(bool)

    def __init__(self, title: str = "Camera Controls", parent=None):
        super().__init__(parent)
        self._body = QtWidgets.QWidget()
        self._body.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self._toggle = QtWidgets.QToolButton(text=title, checkable=True, checked=False)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(QtCore.Qt.RightArrow)
        self._toggle.clicked.connect(self._on_toggle)

        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(self._toggle)
        top.addStretch(1)

        self._wrap = QtWidgets.QWidget()
        wrap_l = QtWidgets.QVBoxLayout(self._wrap)
        wrap_l.setContentsMargins(0, 0, 0, 0)
        wrap_l.addWidget(self._body)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(top)
        lay.addWidget(self._wrap)

        self.set_collapsed(True)

    def set_body_layout(self, layout: QtWidgets.QLayout):
        self._body.setLayout(layout)

    def set_collapsed(self, collapsed: bool):
        self._wrap.setVisible(not collapsed)
        self._toggle.setChecked(not collapsed)
        self._toggle.setArrowType(QtCore.Qt.DownArrow if not collapsed else QtCore.Qt.RightArrow)
        self.toggled.emit(not collapsed)

    def _on_toggle(self):
        self.set_collapsed(self._wrap.isVisible())
