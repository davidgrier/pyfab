#!/usr/bin/env python

"""pyfab.py: GUI for holographic optical trapping."""

from traps import QTrappingPattern
import QFabScreen as fs
from CGH import CGH
from QSLM import QSLM


def main():
    import sys
    from pyqtgraph.Qt import QtGui

    app = QtGui.QApplication(sys.argv)

    fabscreen = fs.QFabScreen(size=(640, 480), gray=True)
    fabscreen.show()

    pattern = QTrappingPattern(fabscreen)

    slm = QSLM()
    cgh = CGH(slm)
    slm.show()

    pattern.pipeline = cgh

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
