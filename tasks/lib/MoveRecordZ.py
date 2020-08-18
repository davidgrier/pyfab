# -*- coding: utf-8 -*-
# MENU: Experiments/Measure z

from ..QTask import QTask
from PyQt5.QtGui import QVector3D
import numpy as np
import os


class MoveRecordZ(QTask):
    """Delay, record, and translate traps in the z direction."""

    def __init__(self, measure_bg=False, **kwargs):
        super(MoveRecordZ, self).__init__(**kwargs)
        self.traps = None
        self.measure_bg = False

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        xc = self.parent.cgh.device.xc
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
            dr = QVector3D(0, 0, dz)
            for n in range(0, 15):
                z_nom = np.absolute(z + dz*n)
                if self.measure_bg:
                    self.register('MoveToCoordinate',
                                  x=self.r_bg[0], y=self.r_bg[1], z=None)
                    self.register('Delay', delay=50)
                    self.register('Record', fn=fn0+'bg_{:03d}.avi'.
                                  format(int(z_nom)), nframes=50)
                    self.register('MoveToCoordinate',
                                  x=self.r[0], y=self.r[1], z=None)
                self.register('Delay', delay=15)
                self.register('Record', fn=fn0+'{:03d}.avi'.
                              format(int(z_nom)),
                              nframes=20)
                self.register('Delay', delay=5)
                self.register('Translate', traps=self.traps, dr=dr)
            # self.register('movetocoordinate')
