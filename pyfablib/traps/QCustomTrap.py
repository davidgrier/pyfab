# -*- coding: utf-8 -*-

"""
QCustomTrap.py: Drawing a trap along a parametric curve.

REFERENCES:
1. José A. Rodrigo, Tatiana Alieva, Eugeny Abramochkin, and Izan Castro   , "Shaping of light beams along curves in three dimensions," Opt.
   Express 21, 20544-20555 (2013)

2. José A. Rodrigo and Tatiana Alieva, "Freestyle 3D laser traps: tools   for studying light-driven particle dynamics and beyond," Optica 2,
   812-815 (2015)
"""

from .QTrap import QTrap
import numpy as np
from numba import njit, prange
from pyqtgraph.Qt import QtGui
from time import time


class QCustomTrap(QTrap):

    def __init__(self, rho=8., m=1, alpha=50, **kwargs):
        super(QCustomTrap, self).__init__(alpha=alpha, **kwargs)
        self._rho = rho
        self._m = m
        self.registerProperty('rho', tooltip=True)
        self.registerProperty('m', decimals=0, tooltip=True)

    def updateStructure(self):
        t0 = time()
        # Allocate integration range
        self.T = 2 * np.pi
        t = np.linspace(0, self.T, 500, endpoint=True)
        # Allocate geometrical buffers
        structure = np.zeros(self.cgh.shape, np.complex_)
        integrand = np.ones((t.size,
                             self.cgh.shape[0],
                             self.cgh.shape[1]),
                            dtype=np.complex_)
        alpha = np.cos(np.radians(self.cgh.phis))
        x = alpha*(np.arange(self.cgh.width) - self.cgh.xs)
        y = np.arange(self.cgh.height) - self.cgh.ys
        xv, yv = np.meshgrid(x, y)
        # Evaluate parameters for integration
        f = self.cgh.focalLength
        lamb = self.cgh.wavelength
        x_0, y_0, z_0 = (self.x_0(t), self.y_0(t), self.z_0(t))
        dx_0, dy_0 = (self.dx_0(t), self.dy_0(t))
        S = self.S(t)
        # Compute integrals for normalization
        S_T = self.S(self.T)
        L = np.trapz(np.sqrt(dx_0**2 + dy_0**2), x=t)
        # Evaluate integrand at all points along the curve
        for idx, ti in enumerate(t):
            self.integrand(ti, xv, yv, S_T, L,
                           self.rho, self.m,
                           f, lamb, x_0[idx], y_0[idx], z_0[idx],
                           S[idx], dx_0[idx], dy_0[idx],
                           integrand[idx])
        # Integrate
        self.integrate(integrand, structure, t, self.cgh.shape)
        self.structure = structure
        print("Time to compute: {}".format(time() - t0))

    def plotSymbol(self):
        sym = QtGui.QPainterPath()
        font = QtGui.QFont('Sans Serif', 10, QtGui.QFont.Black)
        sym.addText(0, 0, font, '+')
        # Scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QtGui.QTransform().scale(scale, scale)
        # Center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2.,
                     - box.y() - box.height()/2.)
        return tr.map(sym)

    @property
    def rho(self):
        '''
        Controls radial scaling of curve.

        The radius of a ring trap in the focal plane
        R = (lambda*f) / (2*pi*rho), where lambda is
        wavelength and f is focal length.
        '''
        return self._rho

    @rho.setter
    def rho(self, rho):
        self._rho = rho
        self.updateStructure()
        self.valueChanged.emit(self)

    @property
    def m(self):
        '''Controls phase gradient along curve.'''
        return self._m

    @m.setter
    def m(self, m):
        self._m = np.int(m)
        self.updateStructure()
        self.valueChanged.emit(self)

    @staticmethod
    @njit(parallel=True, cache=True)
    def integrand(t, x, y, S_T, L, rho, m, f, lamb,
                  x_0, y_0, z_0, S, dx_0, dy_0, buff):
        buff *= np.exp(1.j * (y * x_0 - x * y_0) / rho**2
                       + 1.j * 2*np.pi * m * S / S_T)
        buff *= np.exp(1.j*np.pi * z_0 *
                       ((x - x_0)**2 + (y - y_0)**2)
                       / (lamb * f**2))
        buff *= np.sqrt(dx_0**2 + dy_0**2) / L

    @staticmethod
    @njit(parallel=True, cache=True)
    def integrate(integrand, structure, t, shape):
        nx, ny = shape
        for idx in prange(nx*ny):
            i, j = (idx % (nx-1), idx % (ny-1))
            structure[i, j] = np.trapz(integrand[:, i, j], x=t)

    def S(self, T):
        '''
        Returns the integral of x_0(t)*y_0'(t) - y_0(t)*x_0'(t)
        from 0 to T using the antiderivative
        '''
        pass

    def x_0(self, t):
        '''Component of parametric curve in x direction'''
        pass

    def dx_0(self, t):
        '''Derivative of x_0 with respect to t'''
        pass

    def y_0(self, t):
        '''Component of parametric curve in y direction'''
        pass

    def dy_0(self, t):
        '''Derivative of y_0 with respect to t'''
        pass

    def z_0(self, t):
        '''Component of parametric curve in z direction'''
        pass
