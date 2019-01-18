# -*- coding: utf-8 -*-
# MENU: Motion/Move to focal plane

from .guidedmove import guidedmove
import numpy as np


class movetocoordinate(guidedmove):
    """Can either align many traps in the same line or plane or
    can move one trap to a fixed point.
    By default this task brings all traps to the focal plane."""

    def __init__(self, x=None, y=None, z=0, correct=True, **kwargs):
        super(movetocoordinate, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z
        self.correct = correct

    def calculate_targets(self, traps):
        targets = {}
        for trap in traps.flatten():
            x, y, z = (self.x, self.y, self.z)
            if x is None:
                x = trap.r.x()
            if y is None:
                y = trap.r.y()
            if z is None:
                z = trap.r.z()
            targets[trap] = np.array((x, y, z))
        if self.correct:
            self.register('correct', positions=targets)
        return targets
