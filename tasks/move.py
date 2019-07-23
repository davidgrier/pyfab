# -*- coding: utf-8 -*-

"""Framework for moving all current traps along some trajectory"""

from .task import task
import numpy as np
from PyQt5.QtGui import QVector3D


class move(task):

    def __init__(self, **kwargs):
        super(move, self).__init__(delay=20,
                                   nframes=10**6,
                                   **kwargs)
        self.traps = None
        self.wait = 10
        self.count = self.wait

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        self.trajectories = self.parameterize(self.traps)
        self.N = None
        self.n = 0
        if self.traps.count() > 0:
            if self.trajectories is not None:
                self.N = list(self.trajectories.values())[
                    0].trajectory.shape[0]
                self.n = 0
                self.traps.select(True)
        self.initialize_more(frame)

    def initialize_more(self, frame):
        pass

    def doprocess(self, frame):
        self.process_more(frame)
        if self.count == 0:
            if self.n < self.N:
                new_positions = {}
                traps = self.trajectories.keys()
                for trap in traps:
                    trajectory = self.trajectories[trap].trajectory
                    r = QVector3D(*trajectory[self.n])
                    new_positions[trap] = r
                    trap.moveTo(new_positions[trap])
                self.n += 1
            else:
                self.nframes = 0
            self.count = self.wait
        else:
            self.count -= 1

    def process_more(self, frame):
        pass

    def parameterize(self, traps):
        """
        Returns a dictionary of traps corresponding to their
        respective parameterization.

        Args:
            traps: QTrapGroup of all traps on the QTrappingPattern
        """
        return None


class Trajectory(object):
    '''
    Creates and manipulates a parameterized trajectory in cartesian coordinates
    '''

    def __init__(self, r_i, **kwargs):
        super(Trajectory, self).__init__(**kwargs)
        self.trajectory = np.zeros(shape=(1, 3))
        self.trajectory[0] = np.array([r_i[0], r_i[1], r_i[2]])
        self.last_step = None

    @property
    def r_f(self):
        return self.trajectory[-1]

    @property
    def r_i(self):
        return self.trajectory[0]

    def step(self, d):
        self.last_step = d
        self.trajectory = np.concatenate((self.trajectory,
                                          np.array([self.r_f + d])),
                                         axis=0)

    def stitch(self, trajectory):
        '''Adds another trajectory to the end of the trajectory
        '''
        self.trajectory = np.append(self.trajectory,
                                    trajectory.trajectory, axis=0)

    def __str__(self):
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.2f}".format(x)})
        data = [self.trajectory.shape,
                self.r_i,
                self.r_f,
                self.last_step]
        string = "Trajectory(shape={}, r_i={}, r_f={}, last_step={})"
        return string.format(*data)
