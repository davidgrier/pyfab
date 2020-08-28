# -*- coding: utf-8 -*-
# MENU: Motion/Smooth Translate

from .Move import Move

import numpy as np
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class SmoothTranslate(Move):
    
    def __init__(self, dr=(0., 0., 0.), stepSize=1.0, **kwargs):
        super(SmoothTranslate, self).__init__(stepSize=stepSize, **kwargs)
        self.dr = dr
        
        
    def parameterize(self, traps):
        trajs = []
        dr = np.array(self.dr)
        for trap in traps:
            r0 = [trap.r.x(), trap.r.y(), trap.r.z()]
            npts = int( np.linalg.norm(dr)/self.stepSize )

            traj = [r0 + i*dr/npts for i in range(npts)]
            traj.append(r0+dr)
            print(traj)
            print(np.shape(traj))
            trajs.append(traj)
        self._trajectories = trajs
                    

            