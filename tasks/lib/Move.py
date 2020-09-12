# -*- coding: utf-8 -*-
# MENU: Motion/Move

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
    trajectories: list of (N x 3) - dimensional trajectories
        On frame n, process() moves trap i to trajectories[i][n]
    smooth : bool 
        If True, perform scipy interpolation to smooth trajectories (see 'interpolate' below). Default false.
    stepSize : float
        Size of steps to take each frame after interpolation.
    reverse : 
        If True, move along trajectories in reverse
    Methods
    ------- 
    parameterize : **kwargs (optional)
        Subclass this method to set trajectory as a function of parameters, such as trap position; nframes; etc. 
        See bottom of the file for an example of how to subclass this method. 
     
    interpolate : 
        Smooth trajectories with scipy interpolation.
        - If stepSize is None (default) and nframes=0 (default), the # of points in each trajectory will not change
        - If stepSize is given, then the # of points in each trajectory will scale with length as npts ~ L / stepSize

    '''

    
    def __init__(self, traps=None, trajectories=None, smooth=False, stepSize=None, reverse=False, **kwargs):
        super(Move, self).__init__(**kwargs)
        cgh = self.parent().cgh.device
        self.mpp = cgh.cameraPitch/cgh.magnification  # [microns/pixel]
        self.smooth = smooth
        self.reverse = reverse
        self.stepSize = stepSize      
        self.traps = traps or self.parent().pattern.prev
        self.trajectories = trajectories  
        
    @property
    def group(self):
        if self.traps is not None and len(self.traps) > 0:
            group = self.traps[0].parent()
            while group is not self.parent().pattern:
                if set(group.flatten()) == set(self.traps):
                    return group
                else:
                    group = group.parent()
            print('could not find; making new group...')
            self.parent().pattern.selected = self.traps
            return self.parent().pattern.createGroup()
        else:
            return None
        
    @property
    def traps(self):
        return self._traps
    
    @traps.setter
    def traps(self, traps):
        if traps.__class__.__name__ == 'QTrapGroup':
            self._traps = traps.flatten()
            logger.info('set {} traps'.format(len(traps.flatten())))
            return
        if not isinstance(traps, list):
            traps = [traps]
        if len(traps) > 0 and all([trap.__class__.__name__ =='QTrap' for trap in traps]): 
            self._traps = traps            
        else:
            logger.warning("Setting traps to empty")
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
        if trajectories is None:
            logger.warning('trajectories is None')
            trajectories = []
            
        valid = lambda traj: len(np.shape(traj))==2 and np.shape(traj)[1]==3
        self._trajectories = []
        for i, trap in enumerate(self.traps):
            if i < len(trajectories) and valid( trajectories[i] ):
                self._trajectories.append(np.asarray( trajectories[i] ))
            else:
                self._trajectories.append(np.asarray( [[trap.x, trap.y, trap.z]] ))
                
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
        if self.stepSize is None:    #### If no stepSize is passed, then try to interpolate without changing length of trajectory
            npts = [len(traj) for traj in trajectories]     
        else:
            L = [np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)) for traj in trajectories]
            npts = (np.array(L) * self.mpp / self.stepSize).astype(int)
        trajs = []
        for i, traj in enumerate(trajectories):
            target = traj[-1]
            logger.info('smoothing into {} points'.format(npts[i]))                
            tspace = np.linspace(0, 1, npts[i])
            x = traj[:, 0]
            y = traj[:, 1]
            z = traj[:, 2]
            if npts[i] > 1:
                tck, u = splprep([x, y, z], s=npts[i], k=1)
                xnew, ynew, znew = splev(tspace, tck)
                traj = np.asarray([[xnew[j], ynew[j], znew[j]] for j in range(npts[i])])
                traj[-1] = target
                trajs.append(traj)
        self.trajectories = trajs
        self.npts = npts
    
    def initialize(self, frame):   #### Perform parameterization, interpolate, and preprocess for motion
        logger.info('parameterizing {} traps...'.format(len(self.traps)))
        self.parameterize(self.traps) 
        if self.smooth:                                                                              
            logger.info('smoothing...')                                                                              
            self.interpolate(self.trajectories)
        
        self.setData({'trajectories': [traj.copy() for traj in self.trajectories]})
        self.npts = []
        for i, traj in enumerate(self.trajectories):
            traj = [QVector3D(*point) for point in traj]
            self._trajectories[i] = traj
            self.npts.append(len(traj))
        self.nframes = max(self.npts)*self.skip
        
        logger.info('Parameterized {} trajectories of lengths {}'.format(len(self.npts), self.npts))                            
        logger.info('nframes is {}'.format(self.nframes))
        if self.reverse:
            logger.info('reversing trajectories')
            for traj in self._trajectories:
                traj.reverse()
           
    def process(self, frame):
        logger.info('moving frame {} of {}'.format(self._frame, self.nframes))      
        for i, trap in enumerate(self.traps):
            if self.npts[i]>0:
                trap.moveTo(self.trajectories[i].pop(0))
                self.npts[i] -= 1
                
    def complete(self):
        del self._trajectories  #### So that taskdata isn't overwritten on dequeue
                
        

            