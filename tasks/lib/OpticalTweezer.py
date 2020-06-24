# -*- coding: utf-8 -*-
# MENU: Add trap/Optical tweezer

from ..QTask import QTask
from PyQt5.QtGui import QVector3D


class OpticalTweezer(QTask):
    '''Add an optical tweezer to the trapping pattern'''

    def __init__(self, **kwargs):
        super(OpticalTweezer, self).__init__(**kwargs)

    def complete(self):
        cgh = self.parent().cgh.device
        pos = QVector3D(cgh.xc, cgh.yc, cgh.zc)
        self.parent().pattern.createTrap(pos)
