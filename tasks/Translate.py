# -*- coding: utf-8 -*-

'''Translates traps in some fixed step and direction.'''

from .Task import Task
from PyQt5.QtGui import QVector3D
from pyfablib.traps import QTrap, QTrapGroup


class Translate(Task):

    def __init__(self,
                 traps=None,
                 dr=QVector3D(0, 0, 0),
                 **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.traps = traps
        self.dr = dr

    def initialize(self, frame):
        if self.traps is not None:
            if isinstance(self.traps, QTrapGroup):
                self.traps.select(True)
                self.traps.moveBy(self.dr)
            elif isinstance(self.traps, QTrap):
                self.parent.pattern.pattern.select(True)
                self.traps.moveBy(self.dr)
