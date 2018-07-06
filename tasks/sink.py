# -*- coding: utf-8 -*-
# MENU: Sink

from .assemble import assemble
import numpy as np


class sink(assemble):
    """Demonstration of traps moving into a sink."""

    def __init__(self, **kwargs):
        super(assemble, self).__init__(**kwargs)

    def structure(self, traps):
        vertices = {}
        for trap in traps.flatten():
            vertices[trap] = np.array([320, 240, 50])
        return vertices
