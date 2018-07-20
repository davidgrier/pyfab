# -*- coding: utf-8 -*-
# MENU: Assemble cube

from .assemble import assemble
import numpy as np


class cube(assemble):
    """Demonstration of traps assembling a body-centered cube."""

    def __init__(self, **kwargs):
        super(cube, self).__init__(**kwargs)

    def structure(self, traps):
        if len(traps.flatten()) != 9:
            vertices = None
            print("9 traps are needed for body centered cube")
        else:
            vertices = []
            s = 250
            xc = self.parent.cgh.xc
            yc = self.parent.cgh.yc
            samples = list()
            signs = [1, -1]
            while len(samples) < 4:
                x = [np.random.choice(signs), np.random.choice(signs)]
                if x not in samples:
                    samples.append(x)
            for idx, trap in enumerate(traps.flatten()):
                if idx == 8:
                    vertices.append(np.array([xc, yc, s // 2]))
                else:
                    if idx < 4:
                        z = 0
                    else:
                        z = s
                    i = idx % 4
                    vertices.append(np.array([xc + samples[i][0]*s/2,
                                              yc + samples[i][1]*s/2,
                                              z]))
        return vertices
