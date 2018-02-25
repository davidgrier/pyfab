# -*- coding: utf-8 -*- 

"""QTrap.py: Base class for an optical trap."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from enum import Enum


class state(Enum):
    static = 0
    normal = 1
    selected = 2
    grouping = 3
    inactive = 4


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
                 phi=None,
                 psi=None,
                 state=state.normal,
                 active=True):
        super(QTrap, self).__init__()
        self.active = False
        # organization
        self.parent = parent
        # operational state
        self._state = state
        # appearance
        self.brush = {state.normal: pg.mkBrush(100, 255, 100, 120),
                      state.selected: pg.mkBrush(255, 100, 100, 120),
                      state.grouping: pg.mkBrush(255, 255, 100, 120),
                      state.inactive: pg.mkBrush(0, 0, 255, 120)}
        self.spot = {'pos': QtCore.QPointF(),
                     'size': 10.,
                     'pen': pg.mkPen('k', width=0.5),
                     'brush': self.brush[state],
                     'symbol': 'o'}
        # physical properties
        self.r = r
        self._a = a
        if phi is None:
            self.phi = np.random.uniform(low=0., high=2. * np.pi)
        else:
            self.phi = phi
        # structuring field
        self.psi = psi

        self.active = active

    def moveBy(self, dr):
        """Translate trap.
        """
        self.r = self.r + dr

    def isWithin(self, rect):
        """Return True if this trap lies within the specified rectangle.
        """
        return rect.contains(self.coords())

    def _update(self):
        if self.active:
            self.state = state.selected
            self.parent._update()

    def update_spot(self):
        self.spot['pos'] = self.coords()
        self.spot['size'] = np.clip(10. + self.r.z() / 10., 5., 20.)

    def coords(self):
        """In-plane position of trap for plotting."""
        return self._r.toPointF()

    @property
    def r(self):
        """Three-dimensional position of trap."""
        return self._r

    @r.setter
    def r(self, r):
        active = self.active
        self.active = False
        self._r = QtGui.QVector3D(r)
        self.update_spot()
        self.valueChanged.emit(self)
        self.active = active
        self._update()

    def setX(self, x):
        self._r.setX(x)
        self.update_spot()
        self._update()

    def setY(self, y):
        self._r.setY(y)
        self.update_spot()
        self._update()

    def setZ(self, z):
        self._r.setZ(z)
        self.update_spot()
        self._update()

    def updateAmp(self):
        self.amp = self.a * np.exp(1j * self.phi)
        self._update()

    def setA(self, a):
        self._a = a
        self.updateAmp()

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a):
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
        self.setPhi(phi)
        self.valueChanged.emit(self)

    @property
    def state(self):
        """Current state of trap
        """
        return self._state

    @state.setter
    def state(self, state):
        if self.state is not state.static:
            self._state = state
            self.spot['brush'] = self.brush[state]
