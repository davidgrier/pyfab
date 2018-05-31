# -*- coding: utf-8 -*-

'''Translates traps a step in the z direction'''

from task import task
from pyqtgraph.Qt import QtGui


class translatez(task):

    def __init__(self, traps=None, dr=QtGui.QVector3D(0, 0, 0), **kwargs):
        super(translatez, self).__init__(**kwargs)
        self.traps = traps
	self.dr = dr

    def initialize(self, frame):
        if self.traps is not None:
            self.traps.select(state=True)
            self.traps.moveBy(self.dr)
            self.traps.select(state=False)
