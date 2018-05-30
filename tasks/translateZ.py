# -*- coding: utf-8 -*-

import numpy as np
from task import task
from autotrap import autotrap
from pyqtgraph.Qt import QtGui

class translateZ(autotrap):
    '''Demonstration of translating traps in the z direction by subclassing autotrap.'''
    def __init__(self, **kwargs):
        super(translateZ, self).__init__(nframes=75, delay=100, **kwargs)
        self.dr = QtGui.QVector3D(0, 0, 1)

    def doprocess(self, frame):
        self.traps.select(True)
        self.traps.moveBy(self.dr)
        self.traps.select(False)
