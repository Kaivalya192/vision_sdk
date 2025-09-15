from PyQt5 import QtWidgets


class NetPanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Publisher", parent)
        form = QtWidgets.QFormLayout(self)
        self.ed_ip = QtWidgets.QLineEdit("10.1.156.99")
        self.sp_port = QtWidgets.QSpinBox(); self.sp_port.setRange(1, 65535); self.sp_port.setValue(40001)
        self.chk_publish = QtWidgets.QCheckBox("Publish JSON over UDP"); self.chk_publish.setChecked(True)
        self.sp_cmd_port = QtWidgets.QSpinBox(); self.sp_cmd_port.setRange(1, 65535); self.sp_cmd_port.setValue(40002)
        self.chk_cmd_guard = QtWidgets.QCheckBox("Accept UDP trigger only from receiver IP"); self.chk_cmd_guard.setChecked(True)
        form.addRow("IP", self.ed_ip)
        form.addRow("Port", self.sp_port)
        form.addRow(self.chk_publish)
        form.addRow("Listen port (UDP cmds)", self.sp_cmd_port)
        form.addRow(self.chk_cmd_guard)

