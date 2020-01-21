# -*- coding: utf-8 -*-

"""QTiltedRingTrap.py: Tilted ring using parametric drawing tool."""

from .QCustomTrap import QCustomTrap
import numpy as np


class QTiltedRingTrap(QCustomTrap):

    def __init__(self, s=1., **kwargs):
        super(QTiltedRingTrap, self).__init__(**kwargs)
        self._s = s
        self.registerProperty('s', tooltip=True)

    @property
    def s(self):
        '''Controls axial scaling of curve.'''
        return self._s

    @s.setter
    def s(self, s):
        self._s = s
        self.updateStructure()
        self.valueChanged.emit(self)

    def S(self, T):
        return self.rho**2 * (T - 0)

    def x_0(self, t):
        return self.rho*np.cos(t)

    def dx_0(self, t):
        return - self.rho*np.sin(t)

    def y_0(self, t):
        return self.rho*np.sin(t)

    def dy_0(self, t):
        return self.rho*np.cos(t)

    def z_0(self, t):
        return self.rho*self.s*np.sin(t)
