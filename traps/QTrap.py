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
                 a=None,
                 phi=None,
                 state=states.normal,
                 name=None):
        super(QTrap, self).__init__()
        # organization
        self.parent = parent
        self.name = name
        # physical properties
        self._r = QtGui.QVector3D(0, 0, 0)
        if a is None:
            a = 1.
        if phi is None:
            phi = np.random.uniform() * 2 * np.pi
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
        return rect.contains(self.pos)

    @property
    def r(self):
        """Three-dimensional position of trap."""
        return self._r

    @r.setter
    def r(self, r):
        if r is None:
            return
        elif isinstance(r, QtGui.QVector3D):
            self._r = r
        elif isinstance(r, QtCore.QPointF):
            z = self._r.z()
            self._r = QtGui.QVector3D(r)
            self._r.setZ(z)
        elif isinstance(r, (list, tuple)):
            self._r = QtGui.QVector3D(r[0], r[1], r[2])
        self.valueChanged.emit(self)

    def setA(self, a):
        self.a = a

    def setPhi(self, phi):
        self.phi = phi

    @property
    def pos(self):
        """In-plane position of trap.
        """
        return self.r.toPointF()

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
                'a': self.a,
                'phi': self.phi}
