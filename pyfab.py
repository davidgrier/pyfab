#!/usr/bin/env python

"""pyfab.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
from traps import QTrappingPattern
from QFabGraphicsView import QFabGraphicsView
from QSLM import QSLM
from CGH import CGH


def main():
    import sys

    app = QtGui.QApplication(sys.argv)

    fabscreen = QFabGraphicsView(size=(640, 480), gray=True)
    fabscreen.show()

    pattern = QTrappingPattern(fabscreen)

    slm = QSLM()
    slm.show()

    pattern.pipeline = CGH(slm)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
