#!/usr/bin/env python

"""pyfab.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
from traps import QTrappingPattern
from QFabGraphicsView import QFabGraphicsView
from QSLM import QSLM
from CGH import CGH
import sys


class pyfab(QtGui.QApplication):

    def __init__(self):
        super(pyfab, self).__init__(sys.argv)
        screen_size = (640, 480)
        self.fabscreen = QFabGraphicsView(
            size=screen_size, gray=True, mirrored=False)
        self.fabscreen.show()
        self.pattern = QTrappingPattern(self.fabscreen)
        self.slm = QSLM()
        self.cgh = CGH(self.slm)
        # get calibration constants
        self.cgh.rc = [dim / 2 for dim in screen_size]

        self.pattern.pipeline = self.cgh
        self.fabscreen.sigClosed.connect(self.cleanup)

    def cleanup(self):
        self.slm.close()


if __name__ == '__main__':
    app = pyfab()
    sys.exit(app.exec_())
