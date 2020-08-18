# -*- coding: utf-8 -*-
# MENU: Motion/Assemble lattice

from .Assemble import Assemble
import numpy as np


class Lattice(Assemble):
    """Demonstration of traps assembling in a lattice."""

    def __init__(self, **kwargs):
        super(Lattice, self).__init__(**kwargs)

    def dotask(self):
        if self.assembler.traps is not None:
            # Set tunables
            self.assembler.smooth = True
            self.assembler.stepRate = 5          # [steps/s]
            self.assembler.stepSize = 1.         # [um]
            self.assembler.particleSpacing = 3   # [um]
            self.assembler.gridSpacing = 2       # [um]
            self.assembler.zrange = (5, -50)     # [um]
            self.assembler.tmax = 300            # [steps]
            # Calculate vertices of circle
            vertices = []
            spacing = 10  # um
            ncells = 4
            mpp = self.cgh.cameraPitch / self.cgh.magnification
            xc, yc = (self.cgh.xc, self.cgh.yc)
            xcorner, ycorner, zcorner = (150, 95, 0)
            traps = self.assembler.traps.flatten()
            ntraps = len(traps)
            if ntraps == (ncells+1)**3:
                spacing = spacing / mpp
                x = np.linspace(xcorner, xcorner+spacing*(ncells),
                                ncells+1, endpoint=True)
                y = np.linspace(ycorner, ycorner+spacing*(ncells),
                                ncells+1, endpoint=True)
                z = np.linspace(zcorner, zcorner-spacing*(ncells),
                                ncells+1, endpoint=True)
                xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
                vertices = []
                for i in range(xv.shape[0]):
                    for j in range(xv.shape[1]):
                        for k in range(xv.shape[2]):
                            vertices.append(np.array((xv[i, j, k],
                                                      yv[i, j, k],
                                                      zv[i, j, k])))
                # Set vertices and begin assembly
                self.assembler.targets = vertices
                self.assembler.start()
            else:
                pass
