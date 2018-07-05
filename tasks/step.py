# -*- coding: utf-8 -*-

'''Moves a single QTrap to some location.'''

from .task import task
from pyqtgraph.Qt import QtGui


class step(task):

    def __init__(self, trap=None, r=(0, 0, 0), **kwargs):
        super(step, self).__init__(**kwargs)
        self.trap = trap
        x, y, z = r
        self.r = QtGui.QVector3D(x, y, z)

    def initialize(self, frame):
        if self.trap is not None:
            self.trap.moveTo(self.r)
