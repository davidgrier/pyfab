# -*- coding: utf-8 -*-
# MENU: Switch places

from .assemble import assemble
import numpy as np


class collide(assemble):
    """Try to collide traps and see if they avoid each other."""

    def __init__(self, **kwargs):
        super(collide, self).__init__(**kwargs)

    def structure(self, traps):
        vertices = {}
        trap_list = traps.flatten()
        r_i = np.empty((len(trap_list), 3))
        for idx, trap in enumerate(trap_list):
            r_i[idx] = np.array([trap.r.x(), trap.r.y(), trap.r.z()])
        for idx, trap in enumerate(trap_list):
            vertices[trap] = r_i[(idx + 1) % len(r_i)]
        return vertices
