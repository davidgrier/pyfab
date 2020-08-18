# -*- coding: utf-8 -*-
# MENU: Experiments/Overlap

from .QTask import QTask
import numpy as np



#### No idea what this task does
class Overlap(QTask):
    """Delay, record, and translate traps in the z direction."""

    def __init__(self, traps=None, measure_bg=False, **kwargs):
        super(Overlap, self).__init__(**kwargs)
        self.moving_trap = None
        self.still_trap = None
        self.traps = None

    def initialize(self, frame):
        traps = self.traps or self.parent().pattern.prev
        if len(traps) == 2:
            xc = self.parent.cgh.xc
            d = {np.absolute(xc - traps[0].x): traps[0],
                 np.absolute(xc - traps[1].x): traps[1]}
            self.still_trap = d[min(d.keys())]
            self.moving_trap = d[max(d.keys())]
            self.r = np.array((self.still_trap.r.x(),
                               self.still_trap.r.y()))
            self.r_i = np.array((self.moving_trap.r.x(),
                                 self.moving_trap.r.y()))
            sgn = -1 if self.r_i[0] - self.r[0] > 0 else 1
            self.r_f = np.array((2*self.r[0] - self.r_i[0] + 50*sgn,
                                 self.r_i[1]))

    def process(self):
        if self.still_trap is not None:
            b = self.r_i[1] - self.r[1]
            d = b / 5
            while np.absolute(b) > 5.0:
                self.register('MoveToCoordinate',
                              x=self.r_f[0], y=self.r_f[1], z=None,
                              traps=[self.moving_trap],
                              speed=18.)
                self.register('MoveToCoordinate',
                              x=self.r_i[0], y=self.r_i[1], z=None,
                              traps=[self.moving_trap],
                              speed=18.)
                self.r_i[1] -= d
                self.r_f[1] -= d
                self.register('MoveToCoordinate',
                              x=None, y=self.r_i[1], z=None,
                              traps=[self.moving_trap],
                              speed=18.)
                b = self.r_f[1] - self.r[1]
