# -*- coding: utf-8 -*-

from rendertext import rendertext
from PyQt4 import QtGui


class rendermove(rendertext):
    """Demonstrates trap motion under programmatic control.

    Render the word hello, wait 30 frames, and then move the
    traps horizontally by 10 pixels."""
    
    def __init__(self):
        super(rendermove, self).__init__()
        self.delay = 30

    def dotask(self):
        dr = QtGui.QVector3D(10, 0, 0)
        self.traps.select(True)
        self.traps.moveBy(dr)
        self.traps.select(False)
