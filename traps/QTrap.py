#!/usr/bin/env python

"""QTrap.py: Base class for an optical trap."""

import numpy as np
import pyqtgraph as pg
from PyQt4 import QtCore, QtGui
from states import states


class QTrap(QtCore.QObject):
    """A trap has physical properties, including three-dimensional
    position, relative amplitude and relative phase.
    It also has an appearance as presented on the QFabScreen.
    """

    valueChanged = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self,
                 parent=None,
                 r=None,
                 a=1.,
                 phi=2.*np.pi*np.random.uniform(),
                 state=states.normal,
                 name=None):
        super(QTrap, self).__init__()
        # organization
        self.parent = parent
        self.name = name
        # physical properties
        self.r = r
        self.a = a
        self.phi = phi
        # appearance
        self.symbol = 'o'
        self.brush = {states.normal: pg.mkBrush(100, 255, 100, 120),
                      states.selected: pg.mkBrush(255, 100, 100, 120),
                      states.grouping: pg.mkBrush(255, 255, 100, 120)}
        self.pen = pg.mkPen('k', width=0.5)

        # operational state
        self._state = state

    def moveBy(self, dr):
        """Translate trap.
        """
        self.r = self.r + dr

    def isWithin(self, rect):
        """Return True if this trap lies within the specified rectangle.
        """
        return rect.contains(self.r.toPointF())

    @property
    def r(self):
        """Three-dimensional position of trap."""
        return self._r

    @r.setter
    def r(self, r):
        if r is None:
            self._r = QtGui.QVector(0, 0, 0)
        elif isinstance(r, QtGui.QVector3D):
            self._r = r
        elif isinstance(r, QtCore.QPointF):
            try:
                z = self._r.z()
            except AttributeError:
                z = 0.
            self._r = QtGui.QVector3D(r)
            self._r.setZ(z)
        elif isinstance(r, (list, tuple)):
            self._r = QtGui.QVector3D(r[0], r[1], r[2])
        self.valueChanged.emit(self)

    def updateAmp(self):
        try:
            self.amp = self.a * np.exp(1j * self.phi)
        except AttributeError:
            self.amp = 1. + 0j

    def setA(self, a):
        self._a = a
        self.updateAmp()

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
        if a is not None:
            self.setA(a)
            self.valueChanged.emit(self)

    def setPhi(self, phi):
        self._phi = phi
        self.updateAmp()

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, phi):
        if phi is not None:
            self.setPhi(phi)
            self.valueChanged.emit(self)

    @property
    def state(self):
        """Current state of trap
        """
        return self._state

    @state.setter
    def state(self, state):
        if self.state is not states.static:
            self._state = state

    @property
    def spot(self):
        """Graphical representation of a trap.
        """
        size = np.clip(10. + self.r.z() / 10., 5., 20.)
        return {'pos': self.r.toPointF(),
                'size': size,
                'pen': self.pen,
                'brush': self.brush[self._state],
                'symbol': self.symbol}

    @property
    def properties(self):
        """Physical properties of a trap.
        """
        return {'r': self.r,
                'amp': self.amp}
