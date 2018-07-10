# -*- coding: utf-8 -*-

"""Framework for moving a set of traps to a set of vertices"""

from .parameterize import parameterize, Curve
import numpy as np


class assemble(parameterize):

    def __init__(self, **kwargs):
        super(assemble, self).__init__(**kwargs)

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        self.vertices = self.structure(self.traps)
        self.trajectories = self.parameterize(self.traps,
                                              vertices=self.vertices)

    def parameterize(self, traps, vertices=None):
        '''
        Returns dictionary where Keys are QTraps and Values
        are Curve objects leading to each trap's respective
        vertex.

        Args:
            traps: QTrapGroup of all traps in QTrappingPattern.
        Keywords:
            vertices: Dictionary where Keys are QTraps and Values
                      are 3D ndarray position vectors.
            max_step: Represents maximum velocity. This is the max
                      step size a trap can take between increments.
        '''
        trajectories = None
        if vertices is not None:
            # Initialize trajectories, status, separation
            trajectories = {}
            for trap in traps.flatten():
                r_i = (trap.r.x(), trap.r.y(), trap.r.z())
                trajectories[trap] = Curve(r_i)
                r_v = vertices[trap]
            status, done = self.status(trajectories, vertices)
            # Calculate curves
            while not done:
                # Move each trap a single step
                for trap in trajectories.keys():
                    trajectory = trajectories[trap]
                    if status[trap] is True:
                        # Don't move if it's already there
                        trajectory.step(np.array([0., 0., 0.]))
                    else:
                        # Take a step towards final position
                        r_f = trajectory.r_f
                        r_v = vertices[trap]
                        f_a = self.attraction(r_v - r_f)
                        f_r = self.repulsion(trap, trajectories)
                        trajectory.step(f_a + f_r, scaling=6)
                status, done = self.status(trajectories, vertices)
        return trajectories

    def repulsion(self, trap, trajectories):
        f = np.array([0., 0., 0.])
        q = .1
        for neighbor in trajectories.keys():
            if trap is not neighbor:
                d_i = trajectories[neighbor].r_f - trajectories[trap].r_f
                norm = np.linalg.norm(d_i)
                f += (q**2 / norm**3) * d_i
        return f*-1

    def attraction(self, x):
        k = .01
        f = k * x
        return f

    def structure(self, traps):
        '''
        Returns a dictionary where Keys are QTraps and Values are 
        ndarray cartesian position vectors for vertex location.
        Overwrite in subclass to assemble specific structure.

        Args:
            traps: QTrapGroup of all traps in QTrappingPattern
        '''
        return None

    def status(self, trajectories, vertices):
        '''
        Routine to evaluate whether trajectories have reached
        their respective vertices or not

        Args:
            trajectories: dictionary where Keys are QTraps and Values
                          are Curve objects.
            vertices: dictionary where Keys are QTraps and Values are
                      ndarray cartesian position vectors.
        Returns:
            status: Dictionary where Keys are QTraps and Values are
                    True if trap has reached its vertex and False if not.
            done: True if all traps in status are True, False otherwise
        '''
        status = {}
        done = True
        for trap in trajectories.keys():
            x_v, y_v, z_v = vertices[trap]
            x_f, y_f, z_f = trajectories[trap].r_f
            x_condition = x_v - .5 <= x_f <= x_v + .5
            y_condition = y_v - .5 <= y_f <= y_v + .5
            z_condition = z_v - .5 <= z_f <= z_v + .5
            if x_condition and y_condition and z_condition:
                status[trap] = True
            else:
                status[trap] = False
                done = False
        return status, done
