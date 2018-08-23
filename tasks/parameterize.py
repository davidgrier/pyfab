# -*- coding: utf-8 -*-

"""Framework for moving all current traps along some trajectory"""

from .task import task
import numpy as np
from PyQt4.QtGui import QVector3D


class parameterize(task):

    def __init__(self, **kwargs):
        super(parameterize, self).__init__(delay=100,
                                           skip=3,
                                           nframes=10**6,
                                           **kwargs)
        self.traps = None

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        self.trajectories = self.parameterize(self.traps)
        self.N = None
        self.n = 0
        if self.traps.count() > 0:
            if self.trajectories is not None:
                self.N = list(self.trajectories.values())[0].curve.shape[0]
                self.n = 0
                self.traps.select(True)

    def doprocess(self, frame):
        if self.n < self.N:
            new_positions = {}
            traps = self.trajectories.keys()
            for trap in traps:
                curve = self.trajectories[trap].curve
                r = QVector3D(curve[self.n][0],
                              curve[self.n][1],
                              curve[self.n][2])
                new_positions[trap] = r
                trap.moveTo(new_positions[trap])
            self.n += 1
        else:
            self.nframes = 0

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
        self.last_step = None

    @property
    def r_f(self):
        return self.curve[-1]

    @property
    def r_i(self):
        return self.curve[0]

    def step(self, d):
        self.last_step = d
        self.curve = np.concatenate((self.curve, np.array([self.r_f + d])),
                                    axis=0)

    def __str__(self):
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.2f}".format(x)})
        data = [self.curve.shape,
                self.r_i,
                self.r_f,
                self.last_step]
        string = "Curve(shape={}, r_i={}, r_f={}, last_step={})"
        return string.format(*data)
