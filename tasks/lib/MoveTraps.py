# -*- coding: utf-8 -*-

from ..QTask import QTask
from PyQt5.QtGui import QVector3D
import numpy as np
from scipy.interpolate import splprep, splev
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MoveTraps(QTask):
    '''Move specified traps along specified trajectory. 
    Attributes
    ----------
    traps: list of QTraps       (or QTrapGroup)
        traps to move. If None (default), then use the traps from pyfab's QTrappingPattern
    trajectories: dict with key=QTrap and value=list of tuples (x, y, z)        (or list of list of tuples)
        On frame n, process() moves trap to trajectories[trap][n]
    smooth : bool 
        If True, perform scipy interpolation to smooth trajectories (see 'interpolate' below). Default false.
    stepSize : float
        Size of steps to take each frame (in microns) after interpolation.
        
    Methods
    ------- 
    parameterize : **kwargs (optional)
        Subclass this method to set trajectory as a function of parameters, such as trap position; nframes; etc. Pass additional parameters into 
        To parameterize on __init__ instead, set initialize to pass and uncomment parameterize() in __init__
        See bottom of the file for an example of how to subclass this method. 
     
    interpolate : 
        Smooth trajectories with scipy interpolation.
        - If stepSize is None (default) and nframes=0 (default), the # of points in each trajectory will not change
        - If stepsize is None and nframes is given, all trajectories will contain nframes points after interpolation
        - If stepSize is given, then the # of points in each trajectory will scale with length as npts ~ L / stepSize

    '''

    def __init__(self, traps=None, trajectories={}, smooth=False, stepSize=None, **kwargs):
        super(MoveTraps, self).__init__(**kwargs)
#         self.__dict__.update(kwargs)
        self.smooth = smooth
        self.stepSize = stepSize      
        self.traps = traps or self.parent().pattern.traps     
        self.trajectories = trajectories  
        self.counter = 0
#         self._parameterize()                            #### Uncomment to compute trajectories on __init__ instead of initialize 
        
    @property
    def traps(self):
        return self._traps
    
    @traps.setter
    def traps(self, traps):
        if traps.__class__.__name__ == 'QTrapGroup':
            traps.select(True)
            self._traps = traps.flatten()
            logger.info('trap setter: set {} traps'.format(len(traps.flatten())))
            return
        elif not isinstance(traps, list):
            traps = [traps]
        if all([trap.__class__.__name__ is 'QTrap' for trap in traps]): 
            self._traps = traps
        else:
            logger.warning("elements of trap list must be of type QTrap. Setting to empty")
            self._traps = []

    @property
    def trajectories(self):
        return self._trajectories
    
    @trajectories.setter
    def trajectories(self, trajectories):
        if isinstance(trajectories, list) and len(trajectories) == len(self.traps):
            logger.warning('trajectories passed as list; pairing by index...')
            trajectories = dict(zip(self.traps, trajectories))
        if isinstance(trajectories, dict):
#             print('trajectories set: {}'.format(trajectories))
            for key in trajectories.keys():
                traj = trajectories[key]
                traj = [ traj[i] for i in range(np.shape(traj)[0]) ] if isinstance(traj, np.ndarray) and len(np.shape(traj))==2 else traj
                trajectories[key] = traj
            self._trajectories = trajectories
        else:
            logger.warning('trajectories must be dict or list; setting to empty')
            self.trajectories = [[] for trap in self.traps]

    @property
    def stepSize(self):
        return self._stepSize

    @stepSize.setter
    def stepSize(self, stepSize):
        self._stepSize = stepSize

    def _parameterize(self):
        logger.info('parameterizing {} traps...'.format(len(self.traps)))
        self.parameterize(self.traps)                                                                     #### Note: with new qtask signals, we dont need to know/declare self.nframes
        self.nframes = self.nframes or max( [len(self.trajectories[trap]) for trap in self.traps] ) * self.skip       #### until we run self.process; so we declare it just after we run parametrize()
        logger.info('nframes: {}'.format(self.nframes))
        if self.smooth:                                                                              
            print('smoothing...')                                                                              
            self.interpolate()
        logger.info('Parameterized in {} frames'.format(self.counter))                                 
    
    def parameterize(self, traps):    #### Subclass this method to set trajectories. Must return a dict or list.
        pass
    
#     def interpolate(self):
#         self.nframes = max([len(self.trajectories[trap]) for trap in self.traps])
    def interpolate(self):
        '''
        Smooth out trajectories with scipy interpolation.
        '''
        cgh = self.parent().cgh.device
        mpp = cgh.cameraPitch/cgh.magnification  # [microns/pixel]
        k = self.k if hasattr(self, 'k') else 1
        print('step size: {}'.format(self.stepSize))
        nframes = self.nframes if self.stepSize is None else 0
        for trap in self.traps:
            traj = self.trajectories[trap]
            target = traj[-1]
            data = np.asarray(traj)
            if self.stepSize is None:
                npts = self.nframes or len(traj)
            else:
                L = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                npts = int(L * mpp / self.stepSize)
                print('L: {}'.format(L*mpp))
                nframes = max(nframes, npts)
                logger.info('smoothing into {} points'.format(npts))
            tspace = np.linspace(0, 1, npts)
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            if npts > 1:
                tck, u = splprep([x, y, z], s=npts, k=k)
                xnew, ynew, znew = splev(tspace, tck)
                traj = [(xnew[i], ynew[i], znew[i]) for i in range(npts)]
                traj[-1] = target
                self.trajectories[trap] = traj
                print('set trajectory to interpolated path of length {}'.format(len(traj)))
                self.nframes = nframes*self.skip
    
    def initialize(self, frame):
        logger.info(self._initialized)
        logger.info('counter is {}'.format(self.counter))
        self.counter += 1
        if self.counter == 1:
            self._parameterize()
            save = {}
            for i, key in enumerate(self.trajectories.keys()):
                save[str(i)] = [list(point) for point in self.trajectories[key]]
            with open('trajectories.json', 'w') as f:
                json.dump(save, f)
#             self.setTaskData(self.trajectories)
            self.counter = 0
            self.paths = dict(zip(self.traps, [[] for trap in self.traps]))
            print('init')

    def _process(self, frame):
        logger.info('moving frame {} of {}'.format(self._frame, self.nframes))
        for trap in self.trajectories.keys():
#             print('incrementing traj of len {}'.format(len(self.trajectories[trap])))
#             if len(self.trajectories[trap]) is 0:
#                 return
            print([trap.r.x(), trap.r.y(), trap.z()])
            self.paths[trap].append([trap.r.x(), trap.r.y(), trap.z()])
            pos = self.trajectories[trap].pop(0)
#             if not isinstance(pos, QVector3D) and len(pos) == 3:
#                 pos = QVector3D(*pos)
#             print('moving to {}'.format(pos))
            trap.moveTo(pos)

    def process(self, frame):
        logger.info('moving frame {} of {}'.format(self._frame, self.nframes))
        for trap in self.traps:
#             print([trap.r.x, trap.r.y, trap.z])
            self.paths[trap].append([trap.x, trap.y, trap.z])
#             print('incrementing traj of len {}'.format(len(self.trajectories[trap])))
            if len(self.trajectories[trap]) is 0:
                return
            pos = self.trajectories[trap].pop(0)
            if not isinstance(pos, QVector3D) and len(pos) == 3:
                pos = QVector3D(*pos)
            print('moving to {}'.format(pos))
            trap.moveTo(pos)
	
    def complete(self):
        if True:
            save = {}
            for i, key in enumerate(self.paths.keys()):
                path = self.paths[key]
                path = [path[j] for j in range(np.shape(path)[0])]
                save[str(i)] = [list(point) for point in path]
            with open('paths.json', 'w') as f:
                json.dump(save, f)
 
#         positions = [traj.pop(0) if len(traj)>0 else None for traj in self.trajectories]
#         map(lambda trap, pos: trap.moveTo(pos) if pos is not None else pass, self.traps, positions)


#### Example of how to subclass #####

# class ArcDown(MoveTraps):
    
#     def __init__(self, nframes=20, **kwargs):
#         super(MoveTraps, self).__init__(nframes=nframes, **kwargs)
#         
# 
#     def parameterize(self, traps, radius=30., theta=np.pi/8, **kwargs):      #### Compute trajectories on initialize.         
#         trajs = []                                                   
#         theta_range = np.linspace(0, theta, npts)
#         (xrot, yrot) =  (radius*np.cos(theta_range), radius*np.sin(theta_range))
#         for trap in traps:
#             (x0, y0, z0) = trap.r_p
#             trajs.append([(x0 + xrot[i], y0 + yrot[i], z0) for i in range(npts)])
#         return dict(zip(traps, trajs))
            



