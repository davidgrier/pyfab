# -*- coding: utf-8 -*-
# MENU: Sink

"""Framework for moving a set of traps to a set of vertices"""

from .parameterize import parameterize
import numpy as np


class assemble(parameterize):
    """Demonstration of traps moving into a sink"""

    def __init__(self, **kwargs):
        super(assemble, self).__init__(**kwargs)

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        self.vertices = self.structure(self.traps)
        self.trajectories = self.parameterize(self.traps,
                                              vertices=self.vertices)

    def parameterize(self, traps, vertices=None, max_step=2):
        trajectories = None
        if vertices is not None:
            # Initialize with initial position and velocity
            trajectories = {}
            velocities = {}
            for trap in traps.flatten():
                xi, yi, zi = trap.r.x(), trap.r.y(), trap.r.z()
                trajectories[trap] = np.array([[xi, yi, zi]])
                velocities[trap] = max_step // 2
            # Fill parameterizations
            status = self.status(trajectories, vertices)
            done = False
            while not done:
                for trap in trajectories.keys():
                    # If we're already there append the same position as last
                    if status[trap] is True:
                        trajectories[trap] = np.concatenate((trajectories[trap],
                                                            np.array([trajectories[trap][-1]])),
                                                            axis=0)
                    else:
                        rl = trajectories[trap][-1]
                        rd = vertices[trap]
                        step = ((rd - rl) / np.linalg.norm(rd - rl)) * velocities[trap]
                        
                        trajectories[trap] = np.concatenate((trajectories[trap],
                                                            np.array([trajectories[trap][-1] + step])),
                                                            axis=0)
                # Check if we're done
                status = self.status(trajectories, vertices)
                done = True
                for trap in status.keys():
                    if status[trap] is False:
                        done = False
        return trajectories

    def structure(self, traps):
        '''
        Returns a dictionary where keys are traps and values are vertices.
        Subclass to assemble specific structure.

        Args:
            traps: QTrapGroup of all traps in QTrappingPattern
        '''
        vertices = {}
        for trap in traps.flatten():
            vertices[trap] = np.array([100, 100, 0])
        return vertices

    def status(self, trajectories, vertices):
        """Returns a dictionary where keys are traps and values are
        True if trap has reached its vertex and False if not.
        """
        status = {}
        for trap in trajectories.keys():
            xd, yd, zd = vertices[trap]
            xl, yl, zl = trajectories[trap][-1]
            xl, yl, zl = int(xl), int(yl), int(zl)
            xd, yd, zd = int(xd), int(yd), int(zd)
            if xd == xl and yd == yl and zd == zl:
                status[trap] = True
            else:
                status[trap] = False
        return status

    def detect_collision(self):
        pass

    def avoid(self, collision):
        pass
