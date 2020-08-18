# -*- coding: utf-8 -*-
# MENU: Cull trap

from PyQt5.QtCore import pyqtProperty
from ..QTask import QTask
from pyfablib.traps import QTrapGroup
import random


class CullRandom(QTask):
    """Delete a random trap from a group"""

    def __init__(self, traps=None, **kwargs):
        super(CullRandom, self).__init__(**kwargs)
        self.traps = traps

    @pyqtProperty(object)
    def traps(self):
        return self._traps

    @traps.setter
    def traps(self, traps):
        self._traps = traps or self.parent().pattern.traps

    def complete(self):
        if self.traps is None:
            return
        if isinstance(self.traps, QTrapGroup):
            self.traps = self.traps.flatten()
        trap = random.choice(self.traps)
        self.parent().pattern.clearTrap(trap)
