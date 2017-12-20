from maxtask import maxtask
from PyQt4 import QtGui


class calibrate_cgh(maxtask):

    def __init__(self, **kwargs):
        super(calibrate_cgh, self).__init__(**kwargs)

    def setParent(self, parent):
        self.parent = parent
        self.parent.pattern.clearTraps()
        r = self.parent.cgh.rc + QtGui.QVector3D(100, 0, 0)
        self.parent.pattern.createTrap(r)
        
        
