# -*- coding: utf-8 -*-
# MENU: Auto-Tighten
# VISION: True

from .Task import Task
from PyQt5.QtGui import QVector3D
from collections import deque
import numpy as np
import pandas as pd
import trackpy as tp


class AutoTighten(Task):
    """ Adjust each trap's power output (alpha) until trapped particle's variance approaches a target value """
    
    def __init__(self, target=1, alpha=[0, 10], nframes = 1050, nvar=200, ndelay = 10,  **kwargs): 
        super(AutoTighten, self).__init__(**kwargs)
        self.target = target        ## Target variance
        self.alpha_min = alpha[0]   ## Allowable trap strength (alpha) range
        self.alpha_max = alpha[1]
        self.nframes = nframes      ## Number of frames to run 'doprocess'
        self.nvar = nvar            ## Number of frames to track to compute variance
        self.ndelay = ndelay        ## Number of frames to delay between iterations

    
    def initialize(self, frame):    
        self.vision = self.parent.vision
        self.vision.realTime(True)
        self.count = 0
        self.cdelay=0
        
        
    def doprocess(self, frame):
        if(self.count % self.nvar == 0):  #### First, send trajectories from last nval frames to a dataframe 
            frames = self.vision.video.frames
            nvar = self.nvar
            while(frames[-nvar].framenumber < frame.framenumber - self.nvar): ## Uncomment to use nvar (detecting) VISION frames
                nvar -= 1                                                     ## rather than nvar CAMERA frames.             
            
            d = {'x': [], 'y': [], 'framenumber': []}         
            for frame in frames[-nvar:]:
                for feature in enumerate(frame.features):
                    d['x'].append(feature.model.particle.x_p)
                    d['y'].append(feature.model.particle.y_p)
                    d['framenumber'].append(frame.framenumber)
            trajs = tp.link(pd.DataFrame(data=d), self.vision.linkTol, memory=int(self.vision.nskip+3))
     #### NOTE: Should everything thus far be done using a modified Trajectory object? Or, should we just keep emptying qvision's Video object?
           
        
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
            
            self.traps = self.parent.pattern.pattern     #### Now, find trap positions... NOTE if traps don't move, this can be done in initialize
            for i, trap in enumerate(self.traps.flatten()):
                d['x'].append(trap.r.x)
                d['y'].append(trap.r.y)
                d['val'].append(i)                                                     #### val = trap index 
                d['frame'].append(1)                                                   #### traps at frame 1
            trap_df = pd.DataFrame(data=d)
           
          #### Match trajectories to traps, and adjust each trap based on variance of trapped particle
            pair_df = stat_df.append(trap_df)
            tp.link(pair_df, self.vision.linkTol)
            
            for i, trap in enumerate(self.traps.flatten()):
                particle = pair_df[(pair_df.frame==1) and (pair_df.val==i)].particle   #### traps are frame 1; trap 'val' is trap index
                var = pair_df[(pair_df.frame==0) and (pair_df.particle==particle)].val #### trajs are frame 0, and traj 'val' is variance     
                
                alpha_new = trap.alpha*self.target/var
                if alpha_new > self.alpha_max:
                    trap.alpha(self.alpha_max)
                else if alpha_new < self.alpha_min:
                    trap.alpha(self.alpha_min)
                else:
                    trap.alpha(alpha_new)
           
            self.count += 1
            self.cdelay = self.ndelay    
                
        else if self.cdelay is 0:
           self.count += 1
        else:
            self.cdelay -= 1

            
            
                
            
            
         
            
                    
     
    
