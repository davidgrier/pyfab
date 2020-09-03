# -*- coding: utf-8 -*-
# MENU: Motion/Move To Coordinate

from .Assemble import Assemble

import numpy as np
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class MoveToCoordinate(Assemble):
    ''' Move all selected traps to the specified coordinate, leaving unspecified coordinates unchanged.
         - If 1 coordinate is specified, all traps are moved to a single plane
         - If 2 coordinates are specified, all traps are moved to a single line
         - If 3 coordinates are specified, all traps are moved to a single point
    '''
    
    def __init__(self, x=None, y=None, z=None, smooth=True, stepSize=1.0, **kwargs):
        super(MoveToCoordinate, self).__init__(smooth=smooth, stepSize=stepSize, **kwargs)
        self.x = x
        self.y = y
        self.z = z
        
    def parameterize(self, traps):  #### Bypass path-finding algorithm to move linearly
        r0 = lambda trap: [trap.r.x(), trap.r.y(), trap.r.z()]
        self.trajectories = [np.vstack([r0(trap), self.targets[trap]]) for trap in traps]
        
    def aim(self, traps):
        self.targets = [ [self.x or trap.r.x(), self.y or trap.r.y(), self.z or trap.r.z()] for trap in self.traps]    
        

            
