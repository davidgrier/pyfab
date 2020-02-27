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


class QCustomTrap(QTrap):

    def __init__(self, rho=8., m=1, alpha=1., **kwargs):
        super(QCustomTrap, self).__init__(alpha=alpha, **kwargs)
        self._rho = rho
        self._m = m
        self.registerProperty('rho', decimals=2, tooltip=True)
        self.registerProperty('m', decimals=0, tooltip=True)

    def updateStructure(self):
        # Allocate integration range
        self.T = 2 * np.pi
        npts = 2000
        t = np.linspace(0, self.T, npts, endpoint=True)
        # Get geometrical buffers
        structure, xv, yv = self.getBuffers(t)
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
        self.integrate(t, xv, yv, S_T, L, self.rho, self.m, f, lamb,
                       x_0, y_0, z_0, S, dx_0, dy_0,
                       structure, self.cgh.shape)
        self.structure = structure

    def getBuffers(self, t):
        structure = np.zeros(self.cgh.shape, np.complex_)
        alpha = np.cos(np.radians(self.cgh.phis))
        x = alpha*(np.arange(self.cgh.width) - self.cgh.xs)
        y = np.arange(self.cgh.height) - self.cgh.ys
        xv, yv = np.meshgrid(x, y)
        return structure, xv, yv

    def plotSymbol(self):
        sym = QtGui.QPainterPath()
        font = QtGui.QFont('Sans Serif', 10, QtGui.QFont.Black)
        sym.addText(0, 0, font, '*')
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
    def integrate(t, x, y, S_T, L, rho, m, f, lamb,
                  x_0, y_0, z_0, S, dx_0, dy_0, out, shape):
        nx, ny = shape
        nt = t.size
        dt = (t[-1] - t[0]) / nt
        for idx in prange(nx*ny*nt):
            i, j, k = (idx % (nx-1), idx % (ny-1), idx % (nt-1))
            integrand = np.exp(
                1.j * (y[i, j] * x_0[k] - x[i, j] * y_0[k]) / rho**2
                + 1.j * 2*np.pi * m * S[k] / S_T)
            integrand *= np.exp(
                1.j*np.pi * z_0[k] *
                ((x[i, j] - x_0[k])**2 + (y[i, j] - y_0[k])**2)
                / (lamb * f**2))
            integrand *= np.sqrt(dx_0[k]**2 + dy_0[k]**2) / L
            if (k == 0) or (k == nt-1):
                coeff = 1
            elif k % 2 == 0:
                coeff = 2
            else:
                coeff = 4
            out[i, j] += (dt/3)*coeff*integrand

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
