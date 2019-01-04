# -*- coding: utf-8 -*-
# MENU: Experiments/Measure z

from .task import task
from pyqtgraph.Qt import QtGui
import numpy as np
import os


class moverecordz(task):
    """Delay, record, and translate traps in the z direction."""

    def __init__(self, **kwargs):
        super(moverecordz, self).__init__(**kwargs)
        self.traps = None

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        xc = self.parent.cgh.xc
        trap = self.traps.flatten()[0]
        self.r = np.array((trap.r.x(), trap.r.y()))
        sgn = -1 if self.r[0] - xc > 0 else 1
        self.r_bg = np.array((2*xc - self.r[0] + 50*sgn, self.r[1]))

    def dotask(self):
        self.traps = self.parent.pattern.pattern
        if self.traps.count() > 0:
            fn0, fn_ext = os.path.splitext(self.parent.dvr.filename)
            z = self.traps.r.z()
            dz = -10
            dr = QtGui.QVector3D(0, 0, dz)
            for n in range(0, 50):
                z_nom = np.absolute(z + dz*n)
                self.register('moveto', x=self.r_bg[0], y=self.r_bg[1], z=None)
                self.register('delay', delay=30)
                self.register('record', fn=fn0+'_bg_{:03d}.avi'.
                              format(int(z_nom)))
                self.register('moveto', x=self.r[0], y=self.r[1], z=None)
                self.register('delay', delay=60)
                self.register('record', fn=fn0+'{:03d}.avi'.
                              format(int(z_nom)),
                              nframes=100)
                self.register('translate', traps=self.traps, dr=dr)
