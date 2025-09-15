from PyQt5 import QtWidgets


class ModePanel(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super().__init__("Mode", parent)
        lay = QtWidgets.QHBoxLayout(self)
        self.rad_training = QtWidgets.QRadioButton("Training")
        self.rad_trigger = QtWidgets.QRadioButton("Trigger")
        self.rad_training.setChecked(True)
        self.btn_trigger = QtWidgets.QPushButton("TRIGGER Detect + Publish")
        self.btn_trigger.setEnabled(False)
        lay.addWidget(self.rad_training)
        lay.addWidget(self.rad_trigger)
        lay.addStretch(1)
        lay.addWidget(self.btn_trigger)

