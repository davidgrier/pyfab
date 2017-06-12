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
        self.fabscreen = QFabGraphicsView(size=(640,480), gray=True, mirrored=False)
        self.fabscreen.show()
        self.pattern = QTrappingPattern(self.fabscreen)
        self.slm = QSLM()
        self.slm.show()
        self.pattern.pipeline = CGH(self.slm)
        self.fabscreen.sigFSClosed.connect(self.cleanup)
        self.exec_()

    def cleanup(self):
        self.slm.close()
        

if __name__ == '__main__':
    import sys

    app = pyfab()
