import sys
from PyQt5 import QtWidgets
from app.ui_color_sort.window import ColorSortWindow

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = ColorSortWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
