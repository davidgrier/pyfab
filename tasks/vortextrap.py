# -*- coding: utf-8 -*-
# MENU: Optical vortex

from .task import task
from pyfablib.traps import QVortexTrap


class vortextrap(task):
    """Add an optical vortex to the trapping pattern"""

    def __init__(self, **kwargs):
        super(vortextrap, self).__init__(**kwargs)

    def dotask(self):
        trap = QVortexTrap()
        self.parent.pattern.addTrap(trap)
