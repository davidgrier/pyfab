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
        '''
        trajectories = None
        if vertices is not None:
            # Initialize trajectories, status
            trajectories = {}
            for trap in traps.flatten():
                r_i = (trap.r.x(), trap.r.y(), trap.r.z())
                trajectories[trap] = Curve(r_i)
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
                        dr = self.direct(trap, vertices[trap], trajectories)
                        trajectory.step(dr)
                status, done = self.status(trajectories, vertices)
        return trajectories

    def direct(self, trap, r_v, trajectories):
        '''
        Returns a displacement vector directing trap toward
        its vertex but away from other traps

        Args:
            trap: QTrap of interest
            r_v: ndarray 3D position vector. trap's
                 target position
            trajectories: dictionary where keys are QTraps
                          and values are Curve objects
        '''
        # Initialize variables
        padding = 6
        speed = 5.
        d_v = r_v - trajectories[trap].r_f
        # Direct to vertex
        dx, dy, dz = d_v / np.linalg.norm(d_v)
        # Avoid other particles by simulating spherical force field
        for neighbor in trajectories.keys():
            if trap is not neighbor:
                d_n = trajectories[trap].r_f - trajectories[neighbor].r_f
                r = np.linalg.norm(d_n)
                theta = np.arctan2(d_n[1], d_n[0])
                phi = np.arccos(d_n[2] / r)
                p = padding ** 2
                dx += ((np.cos(theta) * np.sin(phi)) / r) * p
                dy += ((np.sin(theta) * np.sin(phi)) / r) * p
                dz += (np.cos(phi) / r) * p
        # Scale step size by dimensionless speed
        dr = np.array([dx, dy, dz])
        if np.linalg.norm(d_v) < 5.:
            speed = .3
        dr *= speed
        return dr

    def structure(self, traps):
        '''
        Returns a dictionary where keys are QTraps and values are 
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
