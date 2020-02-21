# -*- coding: utf-8 -*-
# MENU: Motion/Move to focal plane

from .motion.GuidedMove import GuidedMove
import numpy as np


class MoveToCoordinate(GuidedMove):
    """Can either align many traps in the same line or plane or
    can move one trap to a fixed point.
    By default this task brings all traps to the focal plane."""

    def __init__(self, x=None, y=None, z=0, correct=False,
                 traps=None, speed=None, **kwargs):
        super(MoveToCoordinate, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z
        self.correct = correct
        self.trap_list = traps
        if speed is not None:
            self.speed = speed

    def calculate_targets(self, traps):
        targets = {}
        if self.trap_list is None:
            self.trap_list = traps.flatten()
        for trap in self.trap_list:
            x, y, z = (self.x, self.y, self.z)
            if x is None:
                x = trap.r.x()
            if y is None:
                y = trap.r.y()
            if z is None:
                z = trap.r.z()
            targets[trap] = np.array((x, y, z))
        if self.correct:
            self.register('Correct', positions=targets)
        return targets
