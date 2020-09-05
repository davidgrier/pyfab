# -*- coding: utf-8 -*-

from ..QTask import QTask
from PyQt5.QtGui import QVector3D
import numpy as np
from scipy.interpolate import splprep, splev
import json
from time import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class Move(QTask):
    '''Move specified traps along specified trajectory. 
    Attributes
    ----------
    traps: list of QTraps       (or QTrapGroup)
        traps to move. If None (default), then use the traps from pyfab's QTrappingPattern
    trajectories: list of list of tuples (x y, z). 
        On frame n, process() moves trap to trajectories[trap][n]
    smooth : bool 
        If True, perform scipy interpolation to smooth trajectories (see 'interpolate' below). Default false.
    stepSize : float
        Size of steps to take each frame (in microns) after interpolation.
        
    Methods
    ------- 
    parameterize : **kwargs (optional)
        Subclass this method to set trajectory as a function of parameters, such as trap position; nframes; etc. 
        See bottom of the file for an example of how to subclass this method. 
     
    interpolate : 
        Smooth trajectories with scipy interpolation.
        - If stepSize is None (default) and nframes=0 (default), the # of points in each trajectory will not change
        - If stepsize is None and nframes is given, all trajectories will contain nframes points after interpolation
        - If stepSize is given, then the # of points in each trajectory will scale with length as npts ~ L / stepSize

    '''

    
    def __init__(self, traps=None, trajectories=None, smooth=False, stepSize=None, **kwargs):
        super(Move, self).__init__(**kwargs)
        cgh = self.parent().cgh.device
        self.mpp = cgh.cameraPitch/cgh.magnification  # [microns/pixel]
        self.smooth = smooth
        self.stepSize = stepSize      
        self.traps = traps or self.parent().pattern.prev
        self.trajectories = trajectories  
 
    @property
    def traps(self):
        return self._traps
    
    @traps.setter
    def traps(self, traps):
        if traps.__class__.__name__ == 'QTrapGroup':
            traps.select(True)
            self._traps = traps.flatten()
            logger.info('set {} traps'.format(len(traps.flatten())))
            return
        elif not isinstance(traps, list):
            traps = [traps]
        if all([trap.__class__.__name__ =='QTrap' for trap in traps]): 
            self._traps = traps
        else:
            logger.warning("elements of trap list must be of type QTrap. Setting to empty")
            self._traps = []

    @property
    def trajectories(self):
        return self._trajectories
    
    @trajectories.setter
    def trajectories(self, trajectories):
        if isinstance(trajectories, dict):
            logger.info('resetting traps using trajectory keys')
            self.traps = list(trajectories.keys())
            trajectories = list(trajectories.values())
        # if not isinstance(trajectories, list):
        #     logger.warning('trajectories must be dict or list; setting to empty')
        #     self._trajectories = [[] for trap in self.traps]
        if trajectories is None:
            logger.warning('trajectories is None; setting to empty')
            self._trajectories = [[] for trap in self.traps]
        elif len(trajectories) != len(self.traps): 
            logger.warning('number of trajectories {} does not match number of traps {}'.format(len(trajectories), len(self.traps)))
            self._trajectories = [traj.copy() for traj in trajectories]
        elif not all( [len(np.shape(traj))==2 and np.shape(traj)[1]==3 for traj in trajectories] ):
            logger.warning('trajectories have wrong dimensions')
        else:
            logger.info('adding {} trajectories'.format(len(trajectories)))
            self._trajectories = trajectories
                
    @property
    def stepSize(self):
        return self._stepSize

    @stepSize.setter
    def stepSize(self, stepSize):
        self._stepSize = stepSize              
    
    def parameterize(self, traps):    #### Subclass this method to set trajectories. Must return a dict or list.
        pass

    def interpolate(self, trajectories):
        '''
        Smooth out trajectories with scipy interpolation.
        '''
        if self.stepSize is None:
            npts = [len(traj) for traj in trajectories]     
        else:
            L = [np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)) for traj in trajectories]
            npts = (np.array(L) * self.mpp / self.stepSize).astype(int)
        self.trajectories = []
        for i, traj in enumerate(trajectories):
            target = traj[-1]
            data = np.asarray(traj)
            logger.info('smoothing into {} points'.format(npts[i]))                
            tspace = np.linspace(0, 1, npts[i])
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            if npts[i] > 1:
                tck, u = splprep([x, y, z], s=npts[i], k=1)
                xnew, ynew, znew = splev(tspace, tck)
                traj = [(xnew[j], ynew[j], znew[j]) for j in range(npts[i])]
                traj[-1] = target
                self.trajectories.append(traj)
        self.npts = npts
    
    def initialize(self, frame):   #### Perform parameterization, interpolate, and preprocess for motion
        logger.info('parameterizing {} traps...'.format(len(self.traps)))
        self.parameterize(self.traps) 
        if self.smooth:                                                                              
            logger.info('smoothing...')                                                                              
            self.interpolate(self.trajectories)
        self.npts = []
        for i, traj in enumerate(self.trajectories):
            traj = [QVector3D(*point) for point in traj]
            self.trajectories[i] = traj
            self.npts.append(len(traj))
        self.nframes = max(self.npts)*self.skip
        logger.info('Parameterized {} trajectories of lengths {}'.format(len(self.npts), self.npts))                            
        logger.info('nframes is {}'.format(self.nframes))
        
        
    def process(self, frame):
        logger.info('moving frame {} of {}'.format(self._frame, self.nframes))      
#         start = time()
        for i, trap in enumerate(self.traps):
            # r = trap.r
            # self.paths[i].append([trap.x, trap.y, trap.z])
            if self.npts[i]>0:
                trap.moveTo(self.trajectories[i].pop(0))
#                 logger.debug('Moved by {}'.format(trap.r.distanceToPoint(r)))
                self.npts[i] -= 1
#         logger.debug('moved in {:03f}'.format(time() - start))
        

            
