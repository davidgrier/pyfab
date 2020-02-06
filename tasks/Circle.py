# -*- coding: utf-8 -*-
# MENU: Motion/Assemble circle

from .motion.GuidedMove import GuidedMove
import numpy as np


class Circle(GuidedMove):
    """Demonstration of traps assembling a circle."""

    def __init__(self, **kwargs):
        super(Circle, self).__init__(**kwargs)

    def calculate_targets(self, traps):
        vertices = []
        radius = 400
        xc = self.parent.cgh.device.xc
        yc = self.parent.cgh.device.yc
        trap_list = traps.flatten()
        for idx, trap in enumerate(trap_list):
            theta = 2*np.pi*(idx+1) / len(trap_list)
            vertices.append(np.array([xc + radius*np.cos(theta),
                                      yc + radius*np.sin(theta),
                                      50]))
        return vertices
