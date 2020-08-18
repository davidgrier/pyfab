# -*- coding: utf-8 -*-

from .Task import Task
from PyQt5.QtGui import QVector3D


class CreateTrap(Task):
    '''Add an optical tweezer to the trapping pattern'''

    def __init__(self, x=100, y=100, z=0, **kwargs):
        super(CreateTrap, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z

    def dotask(self):
        pos = QVector3D(self.x, self.y, self.z)
        self.parent.pattern.createTraps(pos)
        self.done = True
