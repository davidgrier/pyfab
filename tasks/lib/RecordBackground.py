# -*- coding: utf-8 -*-
# MENU: DVR/Record Background

from ..QTask import QTask
import numpy as np


class RecordBackground(QTask):
    """Make particles move in a circle around some point"""
    
    def __init__(self, stepSize=1., **kwargs):
        self.stepSize = stepSize
        super(RecordBackground, self).__init__(**kwargs)
        
    def initialize(self, frame):
        cgh = self.parent().cgh.device
        mpp = cgh.cameraPitch/cgh.magnification  # [microns/pixel]
        stepSize = self.stepSize / mpp
        
        traps = self.parent().pattern.traps.flatten()
        ext = max( [max(trap.x, trap.y) for trap in traps ] ) + 5*stepSize   # Leave room at the edge so particle is actually off screen
        dri = list(np.arange(0, ext, stepSize))    
        trajectories = [ [(trap.x-dr, trap.y-dr, trap.z) for dr in dri] for trap in traps]
        goback = [traj.copy() for traj in trajectories]
        for traj in goback: traj.reverse()
               
        #self.register('Empty', nframes=self.nframes)
        self.register('Move', smooth=False, trajectories=trajectories, filename='1')
        fn = self.parent().dvr.filename
        self.register('Record', nframes=10, fn='/'.join(fn.split('/')[:-1]) + '/background.avi')
        self.parent().dvr.filename = fn
        self.register('Move', smooth=False, trajectories=goback,  filename='2')
