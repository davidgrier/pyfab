# -*- coding: utf-8 -*-
# MENU: Motion/Assemble circle

from .AssembleTraps import AssembleTraps
import numpy as np


class Circle(Assemble):
    """Demonstration of traps assembling a circle."""

    def __init__(self, radius=200, **kwargs):
        super(Circle, self).__init__(**kwargs)

    def aim(self, traps):
        targets = []
        (xc, yc) = (self.cgh.xc, self.cgh.yc)
        N = len(traps)
        theta = 2*np.pi*np.arange(N)/N
        x = xc + self.radius*np.cos(theta)
        y = yc + self.radius*np.sin(theta)
        z = 0
        return [(x[i], y[i], z) for i in range(N)]
   