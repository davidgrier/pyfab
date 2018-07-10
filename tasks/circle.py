# -*- coding: utf-8 -*-
# MENU: Assemble circle

from .assemble import assemble
import numpy as np


class circle(assemble):
    """Demonstration of traps moving into a sink."""

    def __init__(self, **kwargs):
        super(assemble, self).__init__(**kwargs)

    def structure(self, traps):
        vertices = {}
        R = 100
        for idx, trap in enumerate(traps.flatten()):
            theta = idx
            vertices[trap] = np.array([320 + R*np.cos(theta), 240 + R*np.sin(theta), 0])
        return vertices
