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
        
    Methods
    ------- 
    parameterize : **kwargs (optional)
        Subclass this method to set trajectory as a function of parameters, such as trap position; nframe; and additional keywords, if desired.
        See bottom of the file for an example of how to subclass this method. 
     
    interpolate : 
        Smooth trajectories with scipy interpolation.
        scale_length (default False) : if True, resize all trajectories to have self.nframes points
        k : see scipy.splev
        
    process 
        From QTask. On frame j, move traps[i] to position at trajectories[i][j]

    '''

    def __init__(self, traps=None, trajectories=[], smooth=False, **kwargs):
        super(MoveTraps, self).__init__(**kwargs)
        traps = traps or self.parent().pattern.traps   ##.flatten()  ??????
        if  not (nisinstance(traps, list) and isinstance(trajectories, list) ):
            print('Error: traps and trajectories must both be lists')
        for i in range(len(trajectories), len(traps)):
            trajectories.append( [] )
        self.traps = traps
        self.trajectories = trajectories
        self.trajectories = self.parameterize(**kwargs)
        self.nframes = self.nframes or max( [len(traj) for traj in self.trajectories] ) * self.skip
        if smooth:
            self.interpolate(**kwargs)
        
    def parameterize(self, **kwargs):
        pass
    
    def interpolate(self, scale_length=False, k=1, **kwargs):
        '''
        Smooth out trajectories with scipy interpolation.
        '''
        for traj in self.trajectories:
            npts = self.nframes if scale_length else len(traj)
            target = traj[-1]
            data = np.asasarray(traj)
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
    
    def process(self, frame):
        positions = [traj.pop(0) if len(traj)>0 else None for traj in self.trajectories]
        self.pattern.blockRefresh(True)
        map(lambda trap, pos: trap.moveTo(pos) if pos is not None else pass, self.traps, positions)
        self.pattern.blockRefresh(False)
        self.pattern.refresh()







#### Example of how to subclass #####

class ArcDown(MoveTraps):
    
    def __init__(self, nframes=20, **kwargs):
        super(MoveTraps, self).__init__(nframes=nframes, **kwargs)
        

    def parameterize(self, radius=30., theta=np.pi/8, **kwargs):
        self.trajectories = []
        theta_range = np.linspace(0, theta, npts)
        (xrot, yrot) =  (radius*np.cos(theta_range), radius*np.sin(theta_range))
        for trap in self.trap:
            (x0, y0, z0) = trap.r_p
            self.trajectories.append([(x0 + xrot[i], y0 + yrot[i], z0) for i in range(npts)])
            




