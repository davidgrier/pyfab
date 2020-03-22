# -*- coding: utf-8 -*-
# MENU: Motion/Switch places

from .Assemble import Assemble
import numpy as np


class SwitchPlaces(Assemble):
    """Try to collide traps and see if they avoid each other."""

    def __init__(self, **kwargs):
        super(SwitchPlaces, self).__init__(**kwargs)

    def dotask(self):
        if self.assembler.traps is not None:
            # Set tunables
            self.assembler.stepRate = 15        # [steps/s]
            self.assembler.stepSize = .2        # [um]
            self.assembler.particleSpacing = 4  # [um]
            self.assembler.gridSpacing = .5     # [um]
            self.assembler.zrange = (5, -10)    # [um]
            self.assembler.tmax = 300           # [steps]
            # Calculate targets
            traps = self.assembler.traps.flatten()
            targets = {}
            r_i = np.empty((len(traps), 3))
            for idx, trap in enumerate(traps):
                r_i[idx] = np.array([trap.r.x(), trap.r.y(), trap.r.z()])
            for idx, trap in enumerate(traps):
                targets[trap] = r_i[(idx + 1) % len(r_i)]
            # Go!
            self.assembler.targets = targets
            self.assembler.start()
