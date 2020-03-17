# -*- coding: utf-8 -*-

"""
Brownian molecular dynamics simulation for moving
a set of traps to a set of targets
"""

from .TrapMove import TrapMove
import numpy as np
import itertools


class TrapAssemble(TrapMove):

    def __init__(self, **kwargs):
        super(TrapAssemble, self).__init__(**kwargs)
