# -*- coding: utf-8 -*-
# MENU: Random motion

from .task import task
import numpy as np
from collections import OrderedDict


class movealong(task):
    """Demonstration of moving all current traps on some random path."""

    def __init__(self, paths=None, **kwargs):
        super(movealong, self).__init__(**kwargs)
        self.traps = None
        self.trajectories = None
        # paths is a list where each index is an N x 3 array
        self.paths = paths

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern

    def dotask(self):
        if self.traps.count() > 0:
            if self.paths is None:
                # Demonstration of random motion for 200 increments
                paths = []
                trap_list = self.traps.flatten()
                N = 200
                # Initialize list of N x 3 arrays all representating a parameterization
                for idx in range(len(trap_list)):
                    paths.append(np.empty(shape=(N, 3),
                                          dtype=np.float32))
                # Now create parameterizations
                self.trajectories = OrderedDict(zip(trap_list, paths))
                for trap in self.trajectories:
                    trajectory = self.trajectories[trap]
                    # Fill initial position
                    trajectory[0] = np.array([trap.r.x(),
                                              trap.r.y(),
                                              trap.r.z()])
                    # Fill parameterizations
                    for idx, pos in enumerate(trajectory):
                        if idx > 0:
                            x_prev, y_prev, z_prev = trajectory[idx - 1]
                            pos[0], pos[1], pos[2] = (x_prev + np.random.random(),
                                                      y_prev + np.random.random(),
                                                      z_prev + np.random.random())
            else:
                N = self.paths[0].shape[0]
                self.trajectories = OrderedDict(zip(self.traps.flatten(),
                                                    self.paths))
            # Move along paths
            self.traps.select(True)
            for n in range(N):
                self.register('delay', delay=1)
                for trap in self.trajectories:
                    trajectory = self.trajectories[trap]
                    self.register('moveto', trap=trap, r=trajectory[n])
