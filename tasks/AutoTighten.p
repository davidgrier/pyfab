# -*- coding: utf-8 -*-
# MENU: Auto-Trap
# VISION: True

from .Task import Task
from PyQt5.QtGui import QVector3D
from collections import deque
import numpy as np
import pandas as pd
import trackpy as tp
from .Video import Video

N_frames = 200; # number of frames to use to compute variance
## N_it = 10; # number of iterations to perform (This is implicit in nframes in the task "doprocess")
N_delay = 10; # Number of frames to delay between adjustments
alpha_max = 10; 
alpha_min = 0;
tol = None;
count = 0;



class AutoTighten(Task):

    def __init__(self, target=1, alpha=[0, 10], nframes = 1050, nvar=200, ndelay = 10,  **kwargs): 
        super(AutoTighten, self).__init__(**kwargs)
        self.nframes = nframes;     ## Number of frames to run 'doprocess'
        self.nvar = nvar;           ## Number of frames to track to compute variance
        self.ndelay = ndelay;       ## Number of frames to delay between iterations
        self.target = target        ## Target variance
        self.alpha_min = alpha[0]
        self.alpha_max = alpha[1];


    
    def initialize(self, frame):   //initialize stuff, and work out which features are in which traps (pair traps to features)
        self.vision = self.parent.vision
        self.vision.realTime(True)
        self.count = 0;
        self.cdelay=0;
        
        
        
    def process(self, frame):
        self.count += 1;
        if(self.count % self.nvar == 0)
          #### First, make a video from the last nvar frames to compute trajectory 
            var_vid = Video(frames = vision.frames[-nvar:], instrument = self.vision.instrument)  ## Find trajectories for last nvar frames
            
            #### vision.video._frames = vision.video._frames[-nvar:];  ## This is an alternative, but it could with the recording (i.e. vision.video)
            #### var_vid = vision.video;
            
            var_vid.set_trajectories(verbose=False,
                                     search_range=self.vision.linkTol,
                                     memory=int(self.vision.nskip+3))           
            trajs = var_vid.traj_dfs
            
          #### Next, compute each trajectory's variance (to adjust trap strength) and mean position (to pair to trap) 
            d = {'x': [], 'y': [], 'frame': [], 'i': [], 'var': []}
            for i, traj in enumerate(trajs):
                d['x'].append(np.mean(traj.x))
                d['y'].append(np.mean(traj.y))
                d['frame'].append(0)
                d['var'] = np.mean(traj.x.to_numpy()**2 + traj.y.to_numpy**2)
            stat_df = pd.DataFrame(data=d)
          
          #### Use trackpy to match each trajectory with the respective trap
            traps = self.parent.pattern.pattern
            d = {'x': [], 'y': [], 'frame': [], 'i': []}
            for i, trap in enumerate(traps.flatten()):
                d['x'].append(trap.r.x)
                d['y'].append(trap.r.y)
                d['frame'].append(1)
                d['trap'].append(i)
            stat_df.append(pd.DataFrame(data=d))
            tp.link(stat_df, self.vision.linkTol)
            
          #### Adjust each trap based on variance of its trapped particle
            for i, trap in enumerate(traps.flatten()):
                particle = stat_df[stat_df.trap==i].particle
                var = stat_df[stat_df.frame==0 and stat_df.particle==particle].var
                trap.alpha = trap.alpha*self.target/var
            
            self.cdelay = self.ndelay   #### Delay for ndelay frames while traps adjust
                
        if(self.cdelay is not 0):
            self.cdelay -= 1
            self.count -= 1

            
            
                
            
            
         
            
                    
     
    
