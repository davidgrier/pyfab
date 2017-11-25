#!/usr/bin/env python

"""pyfab.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
from traps import QTrappingPattern
from QFabGraphicsView import QFabGraphicsView
from QSLM import QSLM
from CGH import CGH
import sys


class pyfab(QtGui.QWidget):

    def __init__(self):
        super(pyfab, self).__init__()
        screen_size = (640, 480)
        self.fabscreen = QFabGraphicsView(
            size=screen_size, gray=True, mirrored=False)
        self.fabscreen.show()
        self.pattern = QTrappingPattern(self.fabscreen)
        self.slm = QSLM(fake=True)
        self.cgh = CGH(self.slm)
        # get calibration constants
        self.cgh.rc = [dim / 2 for dim in screen_size]

        self.pattern.pipeline = self.cgh
        self.fabscreen.sigClosed.connect(self.cleanup)

    def cleanup(self):
        self.slm.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    win = QtGui.QMainWindow()
    instrument = pyfab()
    instrument.show()
    win.setCentralWidget(instrument)
    sys.exit(app.exec_())
