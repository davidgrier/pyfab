# -*- coding: utf-8 -*-

'''Moves a a group of QTraps to a group of new locations.'''

from .task import task
from pyqtgraph.Qt import QtGui


class correct(task):

    def __init__(self, positions=None, **kwargs):
        super(correct, self).__init__(**kwargs)
        self.positions = positions

    def initialize(self, frame):
        if self.positions is not None:
            for trap in self.positions.keys():
                x, y, z = self.positions[trap]
                trap.moveTo(QtGui.QVector3D(x, y, z))
