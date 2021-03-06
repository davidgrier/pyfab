# -*- coding: utf-8 -*-

"""QBesselIPHTrap.py: Bessel trap using intermediate plane holography"""

from .QTrap import QTrap
import numpy as np
from pyqtgraph.Qt import QtGui
import scipy.integrate as integrate
import scipy.special as special
from scipy import interpolate
import warnings


class QBesselIPHTrap(QTrap):

    def __init__(self, r_alpha=[60.], m=[0.], iz=400., **kwargs):
        super(QBesselIPHTrap, self).__init__(**kwargs)
        self._r_alpha = r_alpha
        self._m = m
        self._iz = iz
        self.lamda = 1.064
        self.registerProperty('r_alpha')
        self.registerProperty('m')
        self.registerProperty('iz')

    def iph_field(self, r, iz, m, r_alpha, lamda, xFactor=1.0):
        """
        Compute electric fields after propagated by a distance z
        via Rayleigh-Sommerfeld approximation.
        Args:
        r: distance from the center.
        z: displacement(s) from the focal plane [pixels].
        m: Angular momentum number
        Ralpha: Radius
        Returns:
        Complex electric field at a point
        """
        warnings.filterwarnings('ignore')
        ci = complex(0., 1.)
        k = 360./lamda
        real = integrate.quadrature(lambda q: q * special.jv(m, q*r_alpha) *
                                    special.jv(m, q*r) *
                                    np.cos(iz * np.sqrt(k**2 - q**2)),
                                    0, xFactor*k, maxiter=100)
        imag = integrate.quadrature(lambda q: q * special.jv(m, q*r_alpha) *
                                    special.jv(m, q*r) *
                                    np.sin(iz * np.sqrt(k**2 - q**2)),
                                    0, xFactor*k, maxiter=100)
        Efield = real[0] + ci * imag[0]
        return Efield

    def updateStructure(self):
        interpN = 450
        qr = self.cgh.qr
        theta = self.cgh.theta
        E_2d = np.zeros(shape=qr.shape, dtype=np.complex_)
        rad = np.linspace(np.amin(qr), np.amax(qr), interpN, endpoint=True)
        ELine = np.zeros(interpN, dtype=np.complex_)
        ci = complex(0., 1.)
        for r_alpha, m in zip(self.r_alpha, self.m):
            for i in range(interpN):
                ELine[i] = self.iph_field(rad[i], self.iz, m,
                                          r_alpha, self.lamda)
            RealESpline = interpolate.splrep(rad, np.real(ELine), s=0)
            ImagESpline = interpolate.splrep(rad, np.imag(ELine), s=0)
            RealE = interpolate.splev(qr, RealESpline, der=0)
            ImagE = interpolate.splev(qr, ImagESpline, der=0)
            E_2d += np.exp(ci*m*theta) * (RealE + ci*ImagE)
        phi = E_2d * np.exp(ci*-1.*(2e5*np.pi*qr**2/(self.lamda*(self.iz*1))))
        self.structure = phi

    def plotSymbol(self):
        sym = QtGui.QPainterPath()
        font = QtGui.QFont('Sans Serif', 14, QtGui.QFont.Black)
        sym.addText(0, 0, font, '*')
        # Scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QtGui.QTransform().scale(scale, scale)
        # Center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @property
    def r_alpha(self):
        return self._r_alpha

    @r_alpha.setter
    def r_alpha(self, r_alpha):
        for idx, r_a in enumerate(r_alpha):
            self._r_alpha[idx] = np.int(r_a)
        self.updateStructure()
        self.valueChanged.emit(self)

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m):
        for idx, m_i in enumerate(m):
            self._m[idx] = np.int(m_i)
        self.updateStructure()
        self.valueChanged.emit(self)

    @property
    def iz(self):
        return self._iz

    @iz.setter
    def iz(self, iz):
        self._iz = np.int(iz)
        self.updateStructure()
        self.valueChanged.emit(self)
