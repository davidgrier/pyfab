# -*- coding: utf-8 -*-

from .Assemble import Assemble
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MoveToCoordinate(Assemble):

    """Moves one selected trap to one point"""

    def __init__(self, x=None, y=None, z=None, **kwargs):
        super(MoveToCoordinate, self).__init__(**kwargs)
        self.x = x
        self.y = y
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
            # Pair
            traps = self.assembler.traps.flatten()
            x, y, z = (self.x, self.y, self.z)
            if len(traps) != 1:
                logger.warning("Select only one trap")
            elif None in [x, y, z]:
                logger.warning("Set desired (x, y, z) position for trap")
            else:
                targets = {traps[0]: np.array((x, y, z))}
                self.assembler.targets = targets
                self.assembler.start()
