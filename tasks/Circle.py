# -*- coding: utf-8 -*-
# MENU: Motion/Assemble circle

from .Assemble import Assemble
import numpy as np


class Circle(Assemble):
    """Demonstration of traps assembling a circle."""

    def __init__(self, **kwargs):
        super(Circle, self).__init__(**kwargs)

    def dotask(self):
        if self.assembler.traps is not None:
            # Set tunables
            self.assembler.stepRate = 3         # [steps/s]
            self.assembler.particleSpacing = 1  # [um]
            self.assembler.gridSpacing = .5     # [um]
            self.assembler.zrange = (5, -10)    # [um]
            self.assembler.tmax = 300           # [steps]
            # Calculate vertices of circle
            vertices = []
            radius = 200  # pixels
            xc, yc = (self.cgh.xc, self.cgh.yc)
            traps = self.assembler.traps.flatten()
            ntraps = len(traps)
            zrange = np.arange(0, 100, 2)
            for idx, trap in enumerate(traps):
                theta = 2*np.pi*(idx+1) / ntraps
                z = np.random.choice(zrange)
                vertices.append(np.array([xc + radius*np.cos(theta),
                                          yc + radius*np.sin(theta),
                                          z]))
            # Set vertices and begin assembly
            self.assembler.targets = vertices
            self.assembler.start()
