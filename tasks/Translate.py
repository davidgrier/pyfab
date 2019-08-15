# -*- coding: utf-8 -*-

'''Translates traps in some fixed step and direction.'''

from .Task import Task
from PyQt5.QtGui import QVector3D


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
            self.traps.select(True)
            self.traps.flatten()[0].moveBy(self.dr)
