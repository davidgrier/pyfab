# -*- coding: utf-8 -*-
# MENU: Motion/Assemble circle

from .Assemble import Assemble
import numpy as np


class Circle(Assemble):
    """Demonstration of traps assembling a circle."""

    def __init__(self, **kwargs):
        super(Circle, self).__init__(**kwargs)
        self.smooth = True
        self.nframes = 300
        self.center = (500, 500)
        self.radius = 200.


    def aim(self, traps):
        (xc, yc) = self.center
        N = len(traps)
        theta = 2*np.pi*np.arange(N)/N
        x = xc + self.radius*np.cos(theta)
        y = yc + self.radius*np.sin(theta)
        z = 0
        self.targets = [(x[i], y[i], z) for i in range(N)]
        print(self.targets)

   
