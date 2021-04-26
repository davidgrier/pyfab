# -*- coding: utf-8 -*-

"""QVortexTrap.py: Optical vortex"""

from .QTrap import QTrap
import numpy as np
from PyQt5.QtCore import pyqtProperty
from PyQt5.QtGui import (QPainterPath, QFont, QTransform)


class QVortexTrap(QTrap):
    """Optical vortex trap

    This is an example of how to subclass QTrap to create a structured
    optical trap.
    1. A structured trap should implement the updateStructure()
    method, which sets the structuring field, self.structure.
    This method will be called whenever properties change in the
    CGH pipeline.
    2. Optionally, the structured trap can override the plotSymbol()
    method, which defines the plot symbol used to represent the trap
    in the trapping pattern.  plotSymbol() can return a QtGuiQPainterPath()
    object, as in this example, or else can return any of the characters
    representing plot symbols for pyqtgraph.  The default symbol is 'o'.
    3. The structured trap has properties that define its structure.
    For an optical vortex, this is the winding number ell.  Routines
    that change these properties should call updateStructure() to
    ensure that the changes take effect.
    """

    def __init__(self, ell=10, **kwargs):
        super(QVortexTrap, self).__init__(**kwargs)
        self._ell = ell  # save private copy in case CGH is not initialized
        self.registerProperty('ell', decimals=0, tooltip=True)

    def updateStructure(self):
        """Helical structuring field defines an optical vortex"""
        self.structure = np.exp((1j * self.ell) * self.cgh.theta)

    def plotSymbol(self):
        """Graphical representation of an optical vortex"""
        sym = QPainterPath()
        # font = QFont('Sans Serif', 10, QFont.Black)
        font = QFont()
        font.setStyleHint(QFont.SansSerif, QFont.PreferAntialias)
        font.setPointSize(10)
        sym.addText(0, 0, font, 'V')
        # scale symbol to unit square
        box = sym.boundingRect()
        scale = 1./max(box.width(), box.height())
        tr = QTransform().scale(scale, scale)
        # center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @pyqtProperty(int)
    def ell(self):
        return self._ell

    @ell.setter
    def ell(self, ell):
        self._ell = np.int(ell)
        self.updateStructure()
