# -*- coding: utf-8 -*-

"""QTiltedRingTrap.py: Tilted ring using parametric drawing tool."""

from .QCustomTrap import QCustomTrap
import numpy as np


class QTiltedRingTrap(QCustomTrap):

    def __init__(self, **kwargs):
        super(QTiltedRingTrap, self).__init__(**kwargs)

    def S(self, T):
        return T - 0

    def x_0(self, t):
        return self.rho * np.cos(t)

    def dx_0(self, t):
        return - self.rho * np.sin(t)

    def y_0(self, t):
        return self.rho * np.sin(t)

    def dy_0(self, t):
        return self.rho * np.cos(t)

    def z_0(self, t):
        return self.rho * np.sin(t)
