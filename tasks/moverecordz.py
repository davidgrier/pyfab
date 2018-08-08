# -*- coding: utf-8 -*-
# MENU: Measure z

from .task import task
from pyqtgraph.Qt import QtGui
import numpy as np
import os


class moverecordz(task):
    """Delay, record, and translate traps in the z direction."""

    def __init__(self, **kwargs):
        super(moverecordz, self).__init__(**kwargs)
        self.traps = None

    def dotask(self):
        self.traps = self.parent.pattern.pattern
        if self.traps.count() > 0:
            fn0, fn_ext = os.path.splitext(self.parent.dvr.filename)
            z = self.traps.r.z()
            dz = -6
            dr = QtGui.QVector3D(0, 0, dz)
            for n in range(0, 50):
                z_nom = np.absolute(z + dz*n)
                self.register('delay', delay=60)
                self.register('record', nframes=100,
                              fn=fn0+'{:03d}.avi'.format(int(z_nom)))
                self.register('translate', traps=self.traps, dr=dr)
