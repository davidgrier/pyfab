# -*- coding: utf-8 -*-

"""Framework for moving all current traps along some trajectory"""

from .task import task
import numpy as np


class parameterize(task):

    def __init__(self, **kwargs):
        super(parameterize, self).__init__(**kwargs)
        self.traps = None

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        self.trajectories = self.parameterize(self.traps)

    def dotask(self):
        if self.traps.count() > 0:
            if self.trajectories is not None:
                # All paths must be same length
                N = list(self.trajectories.values())[0].curve.shape[0]
                # Move along paths
                self.traps.select(True)
                for n in range(N):
                    self.register('delay', delay=1)
                    for trap in self.trajectories:
                        curve = self.trajectories[trap].curve
                        self.register('step', trap=trap, r=curve[n])

    def parameterize(self, traps):
        """
        Returns a dictionary of traps corresponding to their
        respective parameterization.

        Args:
            traps: QTrapGroup of all traps on the QTrappingPattern
        """
        return None


class Curve(object):
    '''
    Creates and manipulates a parameterized curve in cartesian coordinates
    '''

    def __init__(self, r_i, **kwargs):
        super(Curve, self).__init__(**kwargs)
        self.curve = np.zeros(shape=(1, 3))
        self.curve[0] = np.array([r_i[0], r_i[1], r_i[2]])
        self.scaling = None

    @property
    def r_f(self):
        return self.curve[-1]

    @property
    def r_i(self):
        return self.curve[0]

    def step(self, v, scaling=1.0):
        self.scaling = scaling
        step = v * scaling
        self.curve = np.concatenate((self.curve, np.array([self.r_f + step])),
                                    axis=0)

    def __str__(self):
        r_i = self.r_i
        r_f = self.r_f
        print(r_i)
        print(r_f)
        data = [self.curve.shape,
                (int(r_i[0]), int(r_i[1]), int(r_i[2])),
                (int(r_f[0]), int(r_f[1]), int(r_f[2])),
                self.scaling]
        string = "Curve(shape={}, r_i={}, r_f={}, scaling={})"
        return string.format(*data)
