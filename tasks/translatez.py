# -*- coding: utf-8 -*-

'''Translates traps a step in the z direction'''

from task import task
from pyqtgraph.Qt import QtGui


class translatez(task):

    def __init__(self, traps=None, **kwargs):
        super(translatez, self).__init__(**kwargs)
        self.traps = traps

    def initialize(self, frame):
        if self.traps is not None:
            dz = 1
            dr = QtGui.QVector3D(0, 0, dz)
            self.traps.select(True)
            self.traps.moveBy(dr)
            self.traps.select(False)
