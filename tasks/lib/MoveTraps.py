# -*- coding: utf-8 -*-

from ..QTask import QTask
from scipy.interpolate import splprep, splev

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
        self.initialize_parameters(**kwargs)
        self.__dict__.update(kwargs)
        self.smooth = smooth
        self.stepSize = stepSize
        self.traps = traps or self.parent().pattern.traps        
        self._trajectories = {}
        self.trajectories = trajectories
#         self._parameterize()                            #### Uncomment to compute trajectories on __init__ instead of initialize 
        
    @property
    def traps(self):
        return self._traps
    
    @traps.setter
    def traps(self, traps):
        if traps.__class__.__name__ == 'QTrapGroup':
#             traps.select(True)
            self._traps = traps.flatten()
            return
        elif not isinstance(traps, list):
            traps = [traps]
        if all([trap.__class__.name__ is 'QTrap' for trap in traps]): 
            self._traps = traps
        else:
            print("error: elements of trap list must be of type QTrap")
            self._traps = []
    @property
    def trajectories(self):
        return self._trajectories
    
    @trajectories.setter
    def trajectories(self, trajectories):
        if isinstance(trajectories, list) and len(trajectories) == len(self.traps):
            print('Warning: trajectories passed as list; pairing by index...')
            trajectories = dict(zip(self.traps, trajectories))
        if isinstance(trajectories, dict):
            self._trajectories = trajectories
        else:
            print('Warning: trajectories must be dict or list')

    @property
    def stepSize(self):
        return self._stepSize

    @stepSize.setter
    def stepSize(self, stepSize):
        self._stepSize = stepSize

    def _parameterize(self):
        self.trajectories = self.parameterize(self.traps)
        self.nframes = self.nframes or max( [len(traj) for traj in self.trajectories] ) * self.skip  #### Note: with new qtask signals, we dont need to know/declare
        if self.smooth:                                                                              #### self.nframes until we run self.process; so we declare it just after we find trajectories.
            self.interpolate()
    
    def parameterize(self, traps):    #### Subclass this method to declare trajectories. Must return a dict or list.
        return self.trajectories
    
    def interpolate(self):
        '''
        Smooth out trajectories with scipy interpolation.
        '''
        cgh = self.parent().cgh.device
        mpp = cgh.cameraPitch/cgh.magnification  # [microns/pixel]
        stepSize = self.stepSize/mpp if self.stepSize is not None else 
        k = self.k if hasattr(self, 'k') else 1
        for trap in self.traps:
            traj = self.trajectories[trap]
            target = traj[-1]
            data = np.asarray(traj)
            if self.stepSize is None:
                npts = self.nframes or len(traj)
            else:
                L = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
                npts = int(L / stepSize)
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
    
    def initialize(self, frame):
        self._parameterize()
        
    def process(self, frame):
        for trap in self.traps:
            if len(self.trajectories[trap]) > 0:
                trap.moveTo(self.trajectories[trap].pop(0))

     
#         positions = [traj.pop(0) if len(traj)>0 else None for traj in self.trajectories]
# #         self.pattern.blockRefresh(True)
#         map(lambda trap, pos: trap.moveTo(pos) if pos is not None else pass, self.traps, positions)
#         self.pattern.blockRefresh(False)
#         self.pattern.refresh()

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
            




