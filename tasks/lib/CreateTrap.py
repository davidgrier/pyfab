# -*- coding: utf-8 -*-

# MENU: Create Trap

from ..QTask import QTask
from PyQt5.QtGui import QVector3D


class CreateTrap(QTask):
    '''Add an optical tweezer to the trapping pattern'''

    def __init__(self, x=100, y=100, z=0, **kwargs):
        super(CreateTrap, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z

    def complete(self):
        pos = QVector3D(self.x, self.y, self.z)
        self.parent().pattern.createTrap(pos)
