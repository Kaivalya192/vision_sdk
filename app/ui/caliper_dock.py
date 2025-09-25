from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Dict, Any

class CaliperDock(QtWidgets.QDockWidget):
    paramsChanged = QtCore.pyqtSignal(dict)  # Emitted when any parameter changes
    
    def __init__(self, parent=None):
        super().__init__("Caliper Properties", parent)
        self.setObjectName("caliper_dock")
        
        w = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout()
        
        # Length (read-only, shows current length in px and mm)
        self.length_label = QtWidgets.QLabel("0.0 px (0.0 mm)")
        layout.addRow("Length:", self.length_label)
        
        # Thickness spinner (px)
        self.thickness_spin = QtWidgets.QSpinBox()
        self.thickness_spin.setRange(5, 100)
        self.thickness_spin.setValue(15)
        self.thickness_spin.valueChanged.connect(self._emit_params)
        layout.addRow("Thickness (px):", self.thickness_spin)
        
        # Angle spinner (deg)
        self.angle_spin = QtWidgets.QDoubleSpinBox()
        self.angle_spin.setRange(-180, 180)
        self.angle_spin.setDecimals(1)
        self.angle_spin.setSingleStep(1.0)
        self.angle_spin.valueChanged.connect(self._emit_params)
        layout.addRow("Angle (deg):", self.angle_spin)
        
        # Polarity combo
        self.polarity_combo = QtWidgets.QComboBox()
        self.polarity_combo.addItems(["any", "rise", "fall"])
        self.polarity_combo.currentTextChanged.connect(self._emit_params)
        layout.addRow("Edge Polarity:", self.polarity_combo)
        
        # Smoothing kernel size
        self.smooth_spin = QtWidgets.QSpinBox()
        self.smooth_spin.setRange(3, 15)
        self.smooth_spin.setValue(5)
        self.smooth_spin.setSingleStep(2)
        self.smooth_spin.valueChanged.connect(self._emit_params)
        layout.addRow("Smooth:", self.smooth_spin)
        
        # Min contrast threshold
        self.contrast_spin = QtWidgets.QDoubleSpinBox()
        self.contrast_spin.setRange(0, 255)
        self.contrast_spin.setValue(8.0)
        self.contrast_spin.setSingleStep(0.5)
        self.contrast_spin.valueChanged.connect(self._emit_params)
        layout.addRow("Min Contrast:", self.contrast_spin)
        
        # Max number of edges to find
        self.max_edges_spin = QtWidgets.QSpinBox()
        self.max_edges_spin.setRange(1, 4)
        self.max_edges_spin.setValue(2)
        self.max_edges_spin.valueChanged.connect(self._emit_params)
        layout.addRow("Max Edges:", self.max_edges_spin)
        
        # Required number of edges for success
        self.need_edges_spin = QtWidgets.QSpinBox()
        self.need_edges_spin.setRange(1, 2)
        self.need_edges_spin.setValue(1)
        self.need_edges_spin.valueChanged.connect(self._emit_params)
        layout.addRow("Need Edges:", self.need_edges_spin)
        
        # Scale factor (mm/px)
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0.001, 10.0)
        self.scale_spin.setValue(1.0)
        self.scale_spin.setDecimals(4)
        self.scale_spin.setSingleStep(0.001)
        self.scale_spin.valueChanged.connect(self._emit_params)
        layout.addRow("Scale (mm/px):", self.scale_spin)
        
        # Result labels
        self.result_label = QtWidgets.QLabel("")
        layout.addRow("Result:", self.result_label)
        
        w.setLayout(layout)
        self.setWidget(w)
    
    def _emit_params(self, *args):
        """Collect all parameters and emit signal."""
        params = {
            "thickness": self.thickness_spin.value(),
            "angle_deg": self.angle_spin.value(),
            "polarity": self.polarity_combo.currentText(),
            "smooth": self.smooth_spin.value(),
            "min_contrast": self.contrast_spin.value(),
            "max_edges": self.max_edges_spin.value(),
            "need_edges": self.need_edges_spin.value(),
            "px_to_mm": self.scale_spin.value()
        }
        self.paramsChanged.emit(params)
    
    def update_length(self, length_px: float, length_mm: float):
        """Update the length display."""
        self.length_label.setText(f"{length_px:.1f} px ({length_mm:.2f} mm)")
    
    def update_result(self, ok: bool, msg: str):
        """Update the result display."""
        color = "green" if ok else "red"
        self.result_label.setText(f'<span style="color: {color}">{msg}</span>')
        
    def set_scale(self, px_to_mm: float):
        """Set the scale factor."""
        self.scale_spin.setValue(px_to_mm)
    
    def get_params(self) -> Dict[str, Any]:
        """Get current parameter values."""
        return {
            "thickness": self.thickness_spin.value(),
            "angle_deg": self.angle_spin.value(),
            "polarity": self.polarity_combo.currentText(),
            "smooth": self.smooth_spin.value(),
            "min_contrast": self.contrast_spin.value(),
            "max_edges": self.max_edges_spin.value(),
            "need_edges": self.need_edges_spin.value(),
            "px_to_mm": self.scale_spin.value()
        }