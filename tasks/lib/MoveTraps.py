# -*- coding: utf-8 -*-

from ..QTask import QTask
from scipy.interpolate import splprep, splev

class MoveTraps(QTask):
    '''Move specified traps along specified trajectory
    Attributes
    ----------
    traps: list of QTraps
        traps to move. If None (default), then use the traps from pyfab's QTrappingPattern
    trajectories: list of lists of tuples (x, y, z)
        Each trajectory is a list of (n) positions to move trap.   
    smooth : bool 
        If True, perform scipy interpolation to smooth trajectories (see 'interpolate' below). Default false.
    nframes : int
        from QTask. Number of frames to move traps. If nframes=0 (default), use length of trajectory (i.e. move at 30 FPS)
    
    Methods
    ------- 
    parameterize : **kwargs (optional)
        Subclass this method to set trajectory as a function of parameters, such as trap position; nframes; etc. Pass additional parameters into 
        See bottom of the file for an example of how to subclass this method. 
     
    interpolate : 
        Smooth trajectories with scipy interpolation.
        scale_length (default False) : if True, resize all trajectories to have self.nframes points
        k : see scipy.splev
        
    initialize 
        from QTask. To parameterize on __init__ instead, set initialize to pass and uncomment parameterize() in __init__
    process 
        From QTask. On frame j, move traps[i] to position at trajectories[i][j]

    '''

    def __init__(self, traps=None, trajectories=[], smooth=False, **kwargs):
        super(MoveTraps, self).__init__(**kwargs)
        self.__dict__.update(kwargs)
        self.smooth = smooth
        self.traps = traps or self.parent().pattern.traps 
        if isinstance(trajectories, dict):
            self.trajectories = [trajectories[trap] if trap in trajectories.keys() else None for trap in self.traps]
        for i in range(len(trajectories), len(self.traps)):
            trajectories.append( [] )
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
        elif list(traps)[0].__class__.name__ is 'QTrap':
            self._traps = list(traps)
        else:
            print("error: traps must be a list of QTraps")
            self._traps = []
    
    def _parameterize(self):
        self.parameterize()
        self.nframes = self.nframes or max( [len(traj) for traj in self.trajectories] ) * self.skip  #### Note: with new qtask signals, we dont need to know/declare
        if self.smooth:                                                                              #### self.nframes until we run self.process; so we declare it just after we find trajectories.
            self.interpolate(**kwargs)
    
    def parameterize(self):    #### Subclass this method to declare trajectories
        pass
    
    def interpolate(self, scale_length=False, k=1, **kwargs):
        '''
        Smooth out trajectories with scipy interpolation.
        '''
        if hasattr(self, 'scale_length'):
            scale_length = self.scale_length
        if hasattr(self, 'k'):
            k = self.k 
        for traj in self.trajectories:
            npts = self.nframes if scale_length else len(traj)
            target = traj[-1]
            data = np.asarray(traj)
#             L = np.sum(np.linalg.norm(np.diff(traj.data, axis=0), axis=1))
#             npts = int(L / stepSize)
            tspace = np.linspace(0, 1, npts)
            x = data[:, 0]
            y = data[:, 1]
            z = data[:, 2]
            if npts > 1:
                tck, u = splprep([x, y, z], s=npts, k=k)
                xnew, ynew, znew = splev(tspace, tck)
                traj = [(xnew[i], ynew[i], znew[i]) for i in range(npts)]
                traj[-1] = target
    
    def initialize(self, frame):
        self._parameterize
        
    def process(self, frame):
        positions = [traj.pop(0) if len(traj)>0 else None for traj in self.trajectories]
#         self.pattern.blockRefresh(True)
        map(lambda trap, pos: trap.moveTo(pos) if pos is not None else pass, self.traps, positions)
#         self.pattern.blockRefresh(False)
#         self.pattern.refresh()

#### Example of how to subclass #####

# class ArcDown(MoveTraps):
    
#     def __init__(self, nframes=20, **kwargs):
#         super(MoveTraps, self).__init__(nframes=nframes, **kwargs)
        

#     def parameterize(self, radius=30., theta=np.pi/8, **kwargs):      #### Compute trajectories on initialize.         
#     trajs = []                                                   
#         theta_range = np.linspace(0, theta, npts)
#         (xrot, yrot) =  (radius*np.cos(theta_range), radius*np.sin(theta_range))
#         for trap in self.trap:
#             (x0, y0, z0) = trap.r_p
#             trajs.append([(x0 + xrot[i], y0 + yrot[i], z0) for i in range(npts)])
            




