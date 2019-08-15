# -*- coding: utf-8 -*-

'''Moves a a group of QTraps to a group of new locations.'''

from .Task import Task
from PyQt5.QtGui import QVector3D


class Correct(Task):

    def __init__(self, positions=None, **kwargs):
        super(Correct, self).__init__(**kwargs)
        self.positions = positions

    def initialize(self, frame):
        if self.positions is not None:
            for trap in self.positions.keys():
                x, y, z = self.positions[trap]
                trap.moveTo(QVector3D(x, y, z))
