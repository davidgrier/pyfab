# -*- coding: utf-8 -*-
# MENU: Add trap/Bessel IPH trap

from .task import task
from pyfablib.traps.QBesselIPHTrap import QBesselIPHTrap
from pyqtgraph.Qt import QtGui


class bessel_iph(task):
    """Add a bessel trap to the trapping pattern using
    intermediate plane holography"""

    def __init__(self, **kwargs):
        super(bessel_iph, self).__init__(**kwargs)

    def dotask(self):
        xc = self.parent.cgh.xc
        yc = self.parent.cgh.yc
        zc = self.parent.cgh.zc
        trap = QBesselIPHTrap(r=QtGui.QVector3D(xc+0, yc+0, zc))
        self.parent.pattern.addTrap(trap)
