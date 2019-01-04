# -*- coding: utf-8 -*-
# MENU: Motion/Move to point

from .guidedmove import guidedmove
import numpy as np


class movetopoint(guidedmove):
    """Demonstration of one trap moving to a point."""

    def __init__(self, x=0, y=0, z=0, **kwargs):
        super(movetopoint, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z

    def calculate_targets(self, traps):
        targets = {}
        trap = traps.flatten()[0]
        if self.x is None:
            self.x = trap.r.x()
        if self.y is None:
            self.y = trap.r.y()
        if self.z is None:
            self.z = trap.r.z()
        targets[trap] = np.array((self.x, self.y, self.z))
        return targets
