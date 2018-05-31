# -*- coding: utf-8 -*-

'''Translates traps in some fixed step and direction.'''

from task import task
from pyqtgraph.Qt import QtGui


class translate(task):

    def __init__(self, traps=None, dr=QtGui.QVector3D(0, 0, 0), **kwargs):
        super(translate, self).__init__(**kwargs)
        self.traps = traps
        self.dr = dr

    def initialize(self, frame):
        if self.traps is not None:
            self.traps.select(True)
            self.traps.moveBy(self.dr)
            self.traps.select(False)
