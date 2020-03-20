# -*- coding: utf-8 -*-
# MENU: Motion/Assemble cube

from .Assemble import Assemble
import numpy as np


class Cube(Assemble):
    """Demonstration of traps assembling a body centered cube."""

    def __init__(self, **kwargs):
        super(Cube, self).__init__(**kwargs)

    def dotask(self):
        if self.assembler.traps is not None:
            # Set tunables
            self.assembler.stepRate = 3         # [steps/s]
            self.assembler.particleSpacing = 1  # [um]
            self.assembler.gridSpacing = .5     # [um]
            self.assembler.zrange = (5, -10)    # [um]
            self.assembler.tmax = 300           # [steps]
            # Calculate vertices of circle
            traps = self.assembler.traps
            if len(traps.flatten()) != 9:
                vertices = None
                print("9 traps are needed for body centered cube")
            else:
                vertices = []
                s = 200  # [pixels]
                xc = self.parent.cgh.device.xc
                yc = self.parent.cgh.device.yc
                samples = list()
                signs = [1, -1]
                while len(samples) < 4:
                    x = [np.random.choice(signs), np.random.choice(signs)]
                    if x not in samples:
                        samples.append(x)
                for idx, trap in enumerate(traps.flatten()):
                    if idx == 8:
                        vertices.append(np.array([xc, yc, s // 2]))
                    else:
                        if idx < 4:
                            z = 0
                        else:
                            z = s
                        i = idx % 4
                        vertices.append(np.array([xc + samples[i][0]*s/2,
                                                  yc + samples[i][1]*s/2,
                                                  z]))

                # Set vertices and begin assembly
                self.assembler.targets = vertices
                self.assembler.start()
