# -*- coding: utf-8 -*-
# MENU: Motion/Assemble circle

from .guidedmove import guidedmove
import numpy as np


class circle(guidedmove):
    """Demonstration of traps assembling a circle."""

    def __init__(self, **kwargs):
        super(circle, self).__init__(**kwargs)

    def calculate_targets(self, traps):
        vertices = []
        R = 400
        xc = self.parent.cgh.xc
        yc = self.parent.cgh.yc
        trap_list = traps.flatten()
        for idx, trap in enumerate(trap_list):
            theta = 2*np.pi*(idx+1) / len(trap_list)
            vertices.append(np.array([xc + R*np.cos(theta),
                                      yc + R*np.sin(theta),
                                      50]))
        return vertices
