# -*- coding: utf-8 -*-

"""QTrefoilTrap.py: Trefoil knot trap using parametric drawing tool."""

import numpy as np

try:
    from .QCuCustomTrap import QCuCustomTrap as QCustomTrap
    import cupy as cp
    cp.cuda.Device()
except Exception:
    from .QCustomTrap import QCustomTrap


class QTrefoilTrap(QCustomTrap):

    def __init__(self, s=.001, **kwargs):
        super(QTrefoilTrap, self).__init__(**kwargs)
        self._s = s
        self.registerProperty('s', decimals=3, tooltip=True)

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
        return self.rho**2 * (((2./3.)*np.sin(3*T) - 7*T) - 0)

    def x_0(self, t):
        return self.rho * (np.cos(t) - 2*np.cos(2*t))

    def dx_0(self, t):
        return self.rho * (4*np.sin(2*t) - np.sin(t))

    def y_0(self, t):
        return self.rho * (np.sin(t) + 2*np.sin(2*t))

    def dy_0(self, t):
        return self.rho * (np.cos(t) + 4*np.cos(2*t))

    def z_0(self, t):
        return 3 * self.rho * self.s * np.sin(3*t)
