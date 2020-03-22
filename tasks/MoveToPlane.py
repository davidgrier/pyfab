# -*- coding: utf-8 -*-
# MENU: Motion/Move to trapping plane

from .Assemble import Assemble
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MoveToPlane(Assemble):
    """Move traps to desired xy plane. By default moves to z = 0."""

    def __init__(self, z=0, **kwargs):
        super(MoveToPlane, self).__init__(**kwargs)
        self.z = z

    def dotask(self):
        if self.assembler.traps is not None:
            # Set tunables
            self.assembler.smooth = True
            self.assembler.stepRate = 15        # [steps/s]
            self.assembler.stepSize = .2
            self.assembler.particleSpacing = 2  # [um]
            self.assembler.gridSpacing = .5     # [um]
            self.assembler.zrange = (5, -10)    # [um]
            self.assembler.tmax = 300           # [steps]
            # Get and set targets
            traps = self.assembler.traps.flatten()
            targets = {}
            for trap in traps:
                r = (trap.r.x(), trap.r.y(), self.z)
                targets[trap] = np.array(r)
            self.assembler.targets = targets
            # Go!
            self.assembler.start()
