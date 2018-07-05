# -*- coding: utf-8 -*-
# MENU: Random motion

from .parameterize import parameterize
import numpy as np


class random_motion(parameterize):
    """Demonstration of moving all current traps on some random path."""

    def __init__(self, **kwargs):
        super(random_motion, self).__init__(**kwargs)

    def parameterize(self, traps):
        """
        Returns a dictionary of traps corresponding to their
        parameterized random motion

        Args:
            traps: QTrapGroup of all traps on the QTrappingPattern
        """
        # Demonstration of random motion for 200 increments
        paths = []
        trap_list = traps.flatten()
        N = 200
        # Initialize list of N x 3 arrays all representating a parameterization
        for idx in range(len(trap_list)):
            paths.append(np.empty(shape=(N, 3),
                                  dtype=np.float32))
        # Now create parameterizations
        trajectories = dict(zip(trap_list, paths))
        for trap in trajectories.keys():
            trajectory = trajectories[trap]
            # Fill initial position
            trajectory[0] = np.array([trap.r.x(),
                                      trap.r.y(),
                                      trap.r.z()])
            # Fill parameterizations
            for idx, pos in enumerate(trajectory):
                if idx > 0:
                    x_prev, y_prev, z_prev = trajectory[idx - 1]
                    pos[0], pos[1], pos[2] = (x_prev +
                                              np.random.choice([1, -1])*np.random.random(),
                                              y_prev +
                                              np.random.choice([1, -1])*np.random.random(),
                                              z_prev +
                                              np.random.choice([1, -1])*np.random.random())
        return trajectories
