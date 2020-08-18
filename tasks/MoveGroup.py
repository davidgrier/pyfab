# -*- coding: utf-8 -*-

from .Task import Task
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MoveGroup(Task):
    """
    Takes the last QTrapGroup created by user and gives it
    to Pyfab's trap mover (please, help me think of a better
    name than "mover"). Subclass and overwrite calculate_trajectories
    """

    def __init__(self, **kwargs):
        super(MoveGroup, self).__init__(**kwargs)

    def calculate_trajectories(self, traps):
        '''
        Subclass this method to do a cool thing. Right now it just
        initializes Trajectories and doesn't do anything.
        (Check out the code for Trajectory objects, at the
        directory specified in the import!)

        Returns a dictionary, where keys are QTraps
        and values are Trajectory objects
        '''
        trajectories = {}
        for trap in traps.flatten():
            r_i = (trap.r.x(), trap.r.y(), trap.r.z())
            trajectory = np.zeros(shape=(1, 3))
            trajectory[0] = r_i
            # Do something! Perhaps
            # trajectory.data = something (N, 3) shaped
            trajectories[trap] = trajectory
        return trajectories

    def initialize(self, frame):
        '''Makes a user select a TrapGroup to do things to'''
        self.cgh = self.parent.cgh.device
        self.mover = self.parent.mover
        # Set traps from last QTrapGroup created
        pattern = self.parent.pattern.pattern
        group = None
        for child in reversed(pattern.children()):
            if isinstance(child, type(pattern)):
                group = child
                break
        if group is None:
            logger.warning(
                "No traps selected. Please create a QTrapGroup.")
        self.mover.traps = group

    def dotask(self):
        '''
        Set tunables for motion, set calculate_trajectories
        method, and start!
        '''
        # Decide whether to interpolate trajectories and the
        # step size for interpolation. (For this application,
        #  interpolating is not probably not useful)
        self.mover.smooth = False
        self.mover.stepSize = .2   # [um]
        # Step step rate for trap motion
        self.mover.stepRate = 15   # [steps/s]
        # Set mover's general method of trajectory calculation
        # (Help! I can't think of a better name than "mover"!)
        traps = self.mover.traps
        self.mover.trajectories = self.calculate_trajectories(traps)
        # Start moving stuff!
        self.mover.start()
