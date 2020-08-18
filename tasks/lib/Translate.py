# -*- coding: utf-8 -*-
# MENU: Translate

'''Translates traps in some fixed step and direction.'''

from ..QTask import QTask
from PyQt5.QtGui import QVector3D
from pyfablib.traps import QTrap, QTrapGroup


class Translate(QTask):

    def __init__(self, traps=None, dr=(0, 0, 0), **kwargs):
        super(Translate, self).__init__(**kwargs)
        self.traps = traps or self.parent().pattern.prev
        self.dr = dr
                
    def complete(self):
        if self.traps is not None:
            if isinstance(self.traps, QTrapGroup):
                print('group')
                self.traps.select(True)
                self.traps.moveBy(QVector3D(*self.dr))
            elif isinstance(self.traps, QTrap):
                print('loner')
                self.parent().pattern.traps.select(True)
                self.traps.moveBy(QVector3D(*self.dr))
