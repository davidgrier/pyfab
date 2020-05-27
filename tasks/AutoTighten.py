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


class AutoTighten(Task):
    """ 
    
    def __init__(self, target=1, alpha=[0, 10], nframes = 1050, nvar=200, ndelay = 10,  **kwargs): 
        super(AutoTighten, self).__init__(**kwargs)
        self.target = target        ## Target variance
        self.alpha_min = alpha[0]   ## Allowable trap strength (alpha) range
        self.alpha_max = alpha[1]
        self.nframes = nframes      ## Number of frames to run 'doprocess'
        self.nvar = nvar            ## Number of frames to track to compute variance
        self.ndelay = ndelay        ## Number of frames to delay between iterations

    
    def initialize(self, frame):   //initialize stuff, and work out which features are in which traps (pair traps to features)
        self.vision = self.parent.vision
        self.vision.realTime(True)
        self.count = 0
        self.cdelay=0
        
        
        
    def process(self, frame):
        self.count += 1
        if(self.count % self.nvar == 0):  #### First, send trajectories from last nval frames to a dataframe 
            frames = self.vision.video.frames
            nvar = self.nvar
            while(frames[-nvar].framenumber < frame.framenumber - self.nvar and index >= -self.nvar -1): ## Uncomment to use nvar
                nvar += 1                                             ## CAMERA frames, rather than nvar (detecting) VISION frames 
            
            d = {'x': [], 'y': [], 'framenumber': []} 
            for frame in frames[-index:]:
                for feat in enumerate(frame.features):
                    d['x'].append(feat.model.particle.x_p)
                    d['y'].append(feat.model.particle.y_p)
                    d['framenumber'].append(frame.framenumber)
            trajs = tp.link(pd.DataFrame(data=d), self.vision.linkTol, memory=int(self.vision.nskip+3))
            
            #### Next, use mean position to pair particle trajectories with trap positions
            d = {'x': [], 'y': [], 'framenumber': [], 'val': []}
            for particle in range(max(trajs.particle)+1):
                x = trajs[trajs.particle==particle].x
                y = trajs[trajs.particle==particle].y
                d['x'].append(np.mean(x))
                d['y'].append(np.mean(y))
                d['val'].append(np.mean((x - np.mean(x))**2 + (y - np.mean(y))**2))    #### val = variance = |dr|^2        
                d['framenumber'].append(0)                                             #### trajectories at frame 1                
            stat_df = pd.DataFrame(data=d)
            
            traps = self.parent.pattern.pattern     #### Now, find trap positions... NOTE if traps don't move, this can be done in initialize
            for i, trap in enumerate(traps.flatten()):
                d['x'].append(trap.r.x)
                d['y'].append(trap.r.y)
                d['val'].append(i)                                                     #### val = trap index 
                d['frame'].append(1)                                                   #### traps at frame 1
            trap_df = pd.DataFrame(data=d)
           
          #### Match trajectories to traps, and adjust each trap based on variance of trapped particle
            pair_df = stat_df.append(trap_df)
            tp.link(pair_df, self.vision.linkTol)
          
          for particle in range(max(trajs.particle)+1):
                
          
          #### Adjust each trap based on variance of its trapped particle
            for i, trap in enumerate(traps.flatten()):
                particle = stat_df[stat_df.trap==i].particle
                var = stat_df[stat_df.frame==0 and stat_df.particle==particle].var
                trap.alpha(trap.alpha*self.target/var)
            
            self.cdelay = self.ndelay   #### Delay for ndelay frames while traps adjust
                
        if(self.cdelay is not 0):
            self.cdelay -= 1
            self.count -= 1

            
            
                
            
            
         
            
                    
     
    
