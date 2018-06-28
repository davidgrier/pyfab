# -*- coding: utf-8 -*-

"""QVortexTrap.py: Optical vortex"""

from .QTrap import QTrap


class QVortexTrap(QTrap):
    def __init__(self, ell=10, **kwargs):
        super(QVortexTrap, self).__init__(**kwargs)
        self.ell = ell

    def update_field(self):
        if self.cgh is None:
            return
        print('updating:', type(self.cgh))

    @property
    def cgh(self):
        return self._cgh

    @cgh.setter
    def cgh(self, cgh):
        self._cgh = cgh
        self.update_field()