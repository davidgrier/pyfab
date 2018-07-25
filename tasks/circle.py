# -*- coding: utf-8 -*-
# MENU: Assemble circle

from .assemble import assemble
import numpy as np


class circle(assemble):
    """Demonstration of traps assembling a circle."""

    def __init__(self, **kwargs):
        super(circle, self).__init__(**kwargs)

    def structure(self, traps):
        vertices = []
        R = 200
        xc = self.parent.cgh.xc
        yc = self.parent.cgh.yc
        for idx, trap in enumerate(traps.flatten()):
            theta = idx / 1.2
            vertices.append(np.array([xc + R*np.cos(theta),
                                      yc + R*np.sin(theta),
                                      50]))
        return vertices
