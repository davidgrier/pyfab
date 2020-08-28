# -*- coding: utf-8 -*-
# MENU: Motion/Smooth Move To Coordinate

from .Move import Move

import numpy as np
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class SmoothMoveToCoordinate(Move):
    ''' Move all selected traps to the specified coordinate, leaving unspecified coordinates unchanged.
         - If 1 coordinate is specified, all traps are moved to a single plane
         - If 2 coordinates are specified, all traps are moved to a single line
         - If 3 coordinates are specified, all traps are moved to a single point
         
    '''
    
    def __init__(self, x=None, y=None, z=None, stepSize=1.0, **kwargs):
        super(SmoothMoveToCoordinate, self).__init__(stepSize=stepSize, **kwargs)
        self.x = x
        self.y = y
        self.z = z
        
    def parameterize(self, traps):
        
        trajs = []
        for trap in traps:
            
        
            r0 = [trap.r.x(), trap.r.y(), trap.r.z()]
            rf = [self.x or r0[0], self.y or r0[1], self.z or r0[2]]
            print(rf)
            dr = np.array(rf) - r0
            npts = int( np.linalg.norm(dr)/self.stepSize )

            traj = [r0 + i*dr/npts for i in range(npts)]
            traj.append(rf)
            print(traj)
            print(np.shape(traj))
            trajs.append(traj)
        self._trajectories = trajs
                    

            