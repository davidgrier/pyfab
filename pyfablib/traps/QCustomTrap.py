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
from scipy import interpolate
from pyqtgraph.Qt import QtGui
from time import time


class QCustomTrap(QTrap):

    def __init__(self, rho=1., m=1, alpha=50, **kwargs):
        super(QCustomTrap, self).__init__(alpha=alpha, **kwargs)
        self._rho = rho
        self._m = m
        self.registerProperty('rho', tooltip=True)
        self.registerProperty('m', decimals=0, tooltip=True)

    def updateStructure(self):
        start = time()
        # Allocate geometrical buffers
        phi = np.zeros(self.cgh.shape, dtype=np.complex64)
        alpha = np.cos(np.radians(self.cgh.phis))
        x = alpha*(np.arange(self.cgh.width) - self.cgh.xs)
        y = np.arange(self.cgh.height) - self.cgh.ys
        # Allocate integration range
        self.T = 2 * np.pi
        t = np.linspace(0, self.T, 100, endpoint=True)
        # Compute integrals for normalization
        self.S_T = self.S(self.T)
        dr_0 = self.dr_0(t)
        params = interpolate.splrep(t, dr_0, s=0)
        spline = interpolate.BSpline(*params)
        L = spline.integrate(0, self.T)
        # Integrate at each pixel
        for j, yj in enumerate(y):
            for i, xi in enumerate(x):
                integrand = self.integrand(t, xi, yj)
                real_params = interpolate.splrep(t, integrand.real, s=0)
                imag_params = interpolate.splrep(t, integrand.imag, s=0)
                real_spline = interpolate.BSpline(*real_params)
                imag_spline = interpolate.BSpline(*imag_params)
                real = real_spline.integrate(0, self.T)
                imag = imag_spline.integrate(0, self.T)
                phi[j, i] = (real+imag*1.j) / L
                print(j, i)
        print("Time to compute: {}".format(time() - start))
        self.structure = phi

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
        '''Controls radial scaling of curve.'''
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

    def integrand(self, t, x, y):
        '''Integrand for Eq. (6) in Rodrigo (2015)'''
        focalLength = self.cgh.focalLength
        wavelength = self.cgh.wavelength
        phi = np.exp(1.j * np.pi * self.z_0(t) *
                     ((x - self.x_0(t))**2 + (y - self.y_0(t))**2) /
                     (wavelength * focalLength**2))
        Phi = np.exp(1.j * (y * self.x_0(t) - x * self.y_0(t)) / self.rho**2
                     + 1.j * 2*np.pi * self.m * self.S(t) / self.S_T)
        dr_0 = self.dr_0(t)
        return Phi*phi*dr_0

    def S(self, T):
        '''
        Returns the integral of x_0(t)*y_0'(t) - y_0(t)*x_0'(t)
        from 0 to T using the antiderivative
        '''
        pass

    def dr_0(self, t):
        '''Length of tangent to 3D curve projected into z = 0 plane'''
        return np.sqrt(self.dx_0(t)**2 + self.dy_0(t)**2)

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
