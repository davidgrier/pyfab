# -*- coding: utf-8 -*-

"""
QCuCustomTrap.py: Drawing a trap along a parametric curve
                  using CUDA acceleration.
"""

from .QCustomTrap import QCustomTrap
import numpy as np
import cupy as cp


class QCuCustomTrap(QCustomTrap):

    def __init__(self, **kwargs):
        super(QCuCustomTrap, self).__init__(**kwargs)
        self.grid = None
        self.block = None
        self._integrate = cp.RawKernel(r"""
        """, "integrate")

    def getBuffers(self, t):
        structure = cp.zeros(self.cgh.shape, np.complex_)
        integrand = cp.ones((t.size,
                             self.cgh.shape[0],
                             self.cgh.shape[1]),
                            dtype=np.complex_)
        alpha = np.cos(np.radians(self.cgh.phis))
        x = alpha*(np.arange(self.cgh.width) - self.cgh.xs)
        y = np.arange(self.cgh.height) - self.cgh.ys
        x, y = (cp.asarray(x), cp.asarray(y))
        xv, yv = cp.meshgrid(x, y)
        return structure, integrand, xv, yv

    @staticmethod
    def integrand(t, x, y, S_T, L, rho, m, f, lamb,
                  x_0, y_0, z_0, S, dx_0, dy_0, buff):
        buff *= cp.exp(1.j * (y * x_0 - x * y_0) / rho**2
                       + 1.j * 2*np.pi * m * S / S_T)
        buff *= cp.exp(1.j*np.pi * z_0 *
                       ((x - x_0)**2 + (y - y_0)**2)
                       / (lamb * f**2))
        buff *= cp.sqrt(dx_0**2 + dy_0**2) / L
