# -*- coding: utf-8 -*-
# MENU: Motion/Switch places

from .guidedmove import guidedmove
import numpy as np


class switchplaces(guidedmove):
    """Try to collide traps and see if they avoid each other."""

    def __init__(self, **kwargs):
        super(switchplaces, self).__init__(**kwargs)

    def calculate_targets(self, traps):
        targets = {}
        trap_list = traps.flatten()
        r_i = np.empty((len(trap_list), 3))
        for idx, trap in enumerate(trap_list):
            r_i[idx] = np.array([trap.r.x(), trap.r.y(), trap.r.z()])
        for idx, trap in enumerate(trap_list):
            targets[trap] = r_i[(idx + 1) % len(r_i)]
        return targets
