# -*- coding: utf-8 -*-

'''Moves a a group of QTraps to a group of new locations.'''

from .task import task
from pyqtgraph.Qt import QtGui


class relocate(task):

    def __init__(self, new_positions=None, **kwargs):
        super(relocate, self).__init__(**kwargs)
        for trap in new_positions.keys():
            x, y, z = new_positions[trap]
            new_positions[trap] = QtGui.QVector3D(x, y, z)
        self.new_positions = new_positions

    def initialize(self, frame):
        if self.new_positions is not None:
            for trap in self.new_positions.keys():
                trap.moveTo(self.new_positions[trap])
