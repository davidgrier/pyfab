# -*- coding: utf-8 -*-

"""QVortexTrap.py: Optical vortex"""

from .QTrap import QTrap


class QVortexTrap(QTrap):
    def __init__(self, ell=10, **kwargs):
        super(QVortexTrap, self).__init__(**kwargs)
        self.ell = ell
        self.update_field()

    def update_field(self):
        print(type(self.parent))
