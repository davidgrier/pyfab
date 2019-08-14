# -*- coding: utf-8 -*-

"""QUltraTrap.py: Two combined single traps"""

from .QTrap import QTrap
import numpy as np
from pyqtgraph.Qt import QtGui


class QUltraTrap(QTrap):
    """Two combined single traps"""

    def __init__(self, deltaZ=0, deltaPhi=np.pi,**kwargs):
        super(QUltraTrap, self).__init__(**kwargs)
        self._deltaZ = deltaZ  
	# the distance between two single traps (along z) (By default, it is equal to 0.) 
        self._deltaPhi = deltaPhi
	# the phase differnece of two single traps (By default, it is equal to 0.) 
        self.registerProperty('deltaZ', tooltip=True)
        self.registerProperty('deltaPhi', tooltip=True)

    def updateStructure(self):
        """Structuring field"""
        self.structure = -1*np.sin( self.deltaZ/2*(self.cgh.qr)*(self.cgh.qr) + self.deltaPhi/2); 

    def plotSymbol(self):
        """Graphical representation"""
        sym = QtGui.QPainterPath()
        font = QtGui.QFont('Sans Serif', 10, QtGui.QFont.Black)
        sym.addText(0, 0, font, 'U')
        # scale symbol to unit square
        box = sym.boundingRect()
        scale = -1./max(box.width(), box.height())
        tr = QtGui.QTransform().scale(scale, scale)
        # center symbol on (0, 0)
        tr.translate(-box.x() - box.width()/2., -box.y() - box.height()/2.)
        return tr.map(sym)

    @property
    def deltaZ(self):
        return self._deltaZ

    @deltaZ.setter
    def deltaZ(self, deltaZ):
        self._deltaZ = deltaZ
        self.updateStructure()
        self.valueChanged.emit(self)
        
    @property
    def deltaPhi(self):
        return self._deltaPhi

    @deltaPhi.setter
    def deltaPhi(self, deltaPhi):
        self._deltaPhi = deltaPhi
        self.updateStructure()
        self.valueChanged.emit(self)   
