# -*- coding: utf-8 -*-

"""QLineTrap.py: Line Trap"""

from .QTrap import QTrap
import numpy as np
from pyqtgraph.Qt import QtGui


class QLineTrap(QTrap):
    """Optical line trap"""

    def __init__(self, dx=10., dy=0., phi0=0., alpha=10, **kwargs):
        self._dr = QtGui.QVector3D(dx, dy, 0)
        self._phi0 = phi0
        super(QLineTrap, self).__init__(alpha=alpha, **kwargs)
        self.registerProperty('dx', decimals=1, tooltip=True)
        self.registerProperty('dy', decimals=1, tooltip=True)
        self.registerProperty('phi0', decimals=2, tooltip=True)
        self.cgh.sigUpdateTransformationMatrix.connect(self.updateStructure)

    def updateStructure(self):
        """Sinc structuring field defines a line trap"""
        dr = self.cgh.m * self._dr
        self.structure = np.sinc(
            np.add.outer((0.5j * dr.y()) * self.cgh.iqy,
                         (0.5j * dr.x()) * self.cgh.iqx))

    def plotSymbol(self):
        """Graphical representation of a line trap"""
        sym = QtGui.QPainterPath()
        font = QtGui.QFont('Sans Serif', 10, QtGui.QFont.Black)
        sym.addText(0, 0, font, 'L')
        # scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QtGui.QTransform().scale(scale, scale)
        # center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @property
    def dx(self):
        return self._dr.x()

    @dx.setter
    def dx(self, dx):
        self._dr.setX(dx)
        self.updateStructure()
        self.valueChanged.emit(self)

    @property
    def dy(self):
        return self._dr.y()

    @dy.setter
    def dy(self, dy):
        self._dr.setY(dy)
        self.updateStructure()
        self.valueChanged.emit(self)

    @property
    def phi0(self):
        return self._phi0

    @phi0.setter
    def phi0(self, phi0):
        self._phi0 = phi0
        self.updateStructure()
        self.valueChanged.emit(self)
