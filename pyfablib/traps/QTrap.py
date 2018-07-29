# -*- coding: utf-8 -*-

"""QTrap.py: Base class for an optical trap."""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from enum import Enum


class states(Enum):
    static = 0
    normal = 1
    selected = 2
    grouping = 3


class QTrap(QtCore.QObject):
    """A trap has physical properties, including three-dimensional
    position, relative amplitude and relative phase.
    It also has an appearance as presented on the QFabScreen.
    """

    valueChanged = QtCore.pyqtSignal(QtCore.QObject)

    def __init__(self,
                 parent=None,
                 r=QtGui.QVector3D(),
                 a=1.,  # relative amplitude
                 phi=None,  # relative phase
                 psi=None,  # current hologram
                 cgh=None,  # computational pipeline
                 structure=1.+0.j,  # structuring field
                 state=states.normal):
        super(QTrap, self).__init__(parent)

        self.blockRefresh(True)

        # operational state
        self._state = state
        # appearance
        self.brush = {states.normal: pg.mkBrush(100, 255, 100, 120),
                      states.selected: pg.mkBrush(255, 100, 100, 120),
                      states.grouping: pg.mkBrush(255, 255, 100, 120)}
        self.spot = {'pos': QtCore.QPointF(),
                     'size': 10.,
                     'pen': pg.mkPen('k', width=0.5),
                     'brush': self.brush[state],
                     'symbol': self.plotSymbol()}
        # physical properties
        self.properties = dict()
        self.registerProperty('x')
        self.registerProperty('y')
        self.registerProperty('z')
        self.registerProperty('a', decimals=2)
        self.registerProperty('phi', decimals=2)
        self.r = r
        self._a = a
        if phi is None:
            self.phi = np.random.uniform(low=0., high=2. * np.pi)
        else:
            self.phi = phi
        self.psi = psi
        self._structure = structure
        self.cgh = cgh

        self.refreshAppearance()
        self.needsRefresh = True
        self.blockRefresh(False)

    # Customizable methods for subclassed traps
    def plotSymbol(self):
        """Graphical representation of trap"""
        return 'o'

    def refreshAppearance(self):
        """Adapt trap appearance to trap motion and property changes"""
        self.spot['pos'] = self.coords()
        self.spot['size'] = np.clip(10. + self.r.z() / 10., 5., 20.)

    def updateStructure(self):
        """Update structuring field for changes in trap properties
        and calibration constants
        """
        pass

    # Computational pipeline for calculating structure field
    @property
    def cgh(self):
        return self._cgh

    @cgh.setter
    def cgh(self, cgh):
        self._cgh = cgh
        if cgh is None:
            return
        self._cgh.sigUpdateGeometry.connect(self.updateStructure)
        self.updateStructure()

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, field):
        self._structure = self.cgh.bless(field)
        self.refresh()

    # Implementing changes in properties
    def blockRefresh(self, state):
        """Do not send refresh requests to parent if state is True"""
        self._blockRefresh = bool(state)

    def refreshBlocked(self):
        return self._blockRefresh

    def refresh(self):
        """Request parent to implement changes"""
        if self.refreshBlocked():
            return
        self.needsRefresh = True
        self.valueChanged.emit(self)
        self.refreshAppearance()
        self.parent().refresh()

    # Methods for moving the trap
    def moveBy(self, dr):
        """Translate trap by specified displacement vector"""
        self.r = self.r + dr

    def moveTo(self, r):
        """Move trap to position r"""
        self.r = r

    def coords(self):
        """In-plane position of trap for plotting"""
        return self._r.toPointF()

    def isWithin(self, rect):
        """Return True if this trap lies within the specified rectangle"""
        return rect.contains(self.coords())

    # Methods for editing properties with QTrapWidget
    def registerProperty(self, name, decimals=1, tooltip=False):
        """Register a property so that it can be edited"""
        self.properties[name] = {'decimals': decimals, 'tooltip': tooltip}

    @QtCore.pyqtSlot(str, float)
    def setProperty(self, name, value):
        """Thread-safe method to change a specified property without
        emitting signals.  This is called by QTrapWidget when the
        user edits a property.  Blocking signals prevents a loop.
        """
        self.blockSignals(True)
        setattr(self, name, value)
        self.blockSignals(False)

    # Trap properties
    @property
    def r(self):
        """Three-dimensional position of trap"""
        return self._r

    @r.setter
    def r(self, r):
        self._r = QtGui.QVector3D(r)
        self.refresh()

    @property
    def x(self):
        return self._r.x()

    @x.setter
    def x(self, x):
        self._r.setX(x)
        self.refresh()

    @property
    def y(self):
        return self._r.y()

    @y.setter
    def y(self, y):
        self._r.setY(y)
        self.refresh()

    @property
    def z(self):
        return self._r.z()

    @z.setter
    def z(self, z):
        self._r.setZ(z)
        self.refresh()

    @property
    def a(self):
        """Relative amplitude of trap"""
        return self._a

    @a.setter
    def a(self, a):
        self._a = a
        self.amp = a * np.exp(1j * self.phi)
        self.refresh()

    @property
    def phi(self):
        """Relative phase of trap"""
        return self._phi

    @phi.setter
    def phi(self, phi):
        self._phi = phi
        self.amp = self.a * np.exp(1j * phi)
        self.refresh()

    @property
    def state(self):
        """Current state of trap"""
        return self._state

    @state.setter
    def state(self, state):
        if self.state is not states.static:
            self._state = state
            self.spot['brush'] = self.brush[state]
