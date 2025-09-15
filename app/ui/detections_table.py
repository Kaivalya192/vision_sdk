from typing import Dict, List, Optional
from PyQt5 import QtWidgets


class DetectionsTable(QtWidgets.QTableWidget):
    def __init__(self, parent=None):
        super().__init__(0, 8, parent)
        self.setHorizontalHeaderLabels(["Obj", "Inst", "x", "y", "θ(deg)", "score", "inliers", "center"])
        self.horizontalHeader().setStretchLastSection(True)

    def populate(self, objects_report: List[Dict]):
        rows = sum(len(o.get("detections", [])) for o in objects_report)
        self.setRowCount(rows)
        r = 0
        for obj in objects_report:
            name = obj.get("name", "?")
            for det in obj.get("detections", []):
                ctr = det.get("center")
                vals = [
                    name,
                    str(det.get("instance_id", 0)),
                    f"{det['pose']['x']:.1f}",
                    f"{det['pose']['y']:.1f}",
                    f"{det['pose']['theta_deg']:.1f}",
                    f"{det.get('score', 0.0):.3f}",
                    str(det.get('inliers', 0)),
                    f"({int(ctr[0])},{int(ctr[1])})" if ctr else "?",
                ]
                for c, text in enumerate(vals):
                    self.setItem(r, c, QtWidgets.QTableWidgetItem(text))
                r += 1
        if rows == 0:
            self.setRowCount(0)

