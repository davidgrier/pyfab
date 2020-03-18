# -*- coding: utf-8 -*-
# MENU: Motion/Assemble circle

from .Task import Task
import numpy as np


class Circle(Task):
    """Demonstration of traps assembling a circle."""

    def __init__(self, **kwargs):
        super(Circle, self).__init__(**kwargs)

    def dotask(self):
        cgh = self.parent.cgh.device
        pattern = self.parent.pattern.pattern
        group = None
        for child in reversed(pattern.children()):
            if isinstance(child, type(pattern)):
                group = child
                break
        assembler = self.parent.assembler
        # Set traps from last QTrapGroup
        assembler.traps = group
        # Calculate vertices of circle
        vertices = []
        radius = 400  # pixels
        xc, yc = (cgh.xc, cgh.yc)
        traps = assembler.traps.flatten()
        ntraps = len(traps)
        for idx, trap in enumerate(traps):
            theta = 2*np.pi*(idx+1) / ntraps
            vertices.append(np.array([xc + radius*np.cos(theta),
                                      yc + radius*np.sin(theta),
                                      50]))
        assembler.targets = vertices
        assembler.start()
