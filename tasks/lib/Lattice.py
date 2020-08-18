# -*- coding: utf-8 -*-
# MENU: Motion/Assemble lattice

from .Assemble import Assemble
import numpy as np


class Lattice(Assemble):
    """Demonstration of traps assembling in a lattice."""

    def __init__(self, center2=None, spacing=3, ncells=1, corner3=(150, 95, 0), **kwargs):
        super(Lattice, self).__init__(**kwargs)
        self.spacing = spacing  #um
        self.ncells = ncells
        self.center2 = center2 or (self.parent().cgh.device.xc, self.parent().cgh.device.yc) 
        self.corner3 = corner3

#     def dotask(self):
#         if self.assembler.traps is not None:
#             # Set tunables
#             self.assembler.smooth = True
#             self.assembler.stepRate = 5          # [steps/s]
#             self.assembler.stepSize = 1.         # [um]
#             self.assembler.particleSpacing = 3   # [um]
#             self.assembler.gridSpacing = 2       # [um]
#             self.assembler.zrange = (5, -50)     # [um]
#             self.assembler.tmax = 300            # [steps]
            # Calculate vertices of circle
    def aim(self, traps):
        vertices = []
        spacing = self.spacing  
        ncells = self.ncells
        mpp = self.parent().cgh.device.cameraPitch / self.parent().cgh.device.magnification
        (xc, yc) = self.center2
        (xcorner, ycorner, zcorner) = self.corner3
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
            self.targets = vertices
#             print(self.targets)
        else:
            print('aim failed')
                
