# -*- coding: utf-8 -*-
# MENU: Experiments/Record Background

# from .QExperiment import QExperiment
# import numpy as np


# class RecordBackground(QExperiment):
            
#     def initialize(self, frame):
#         self.info = 'RecordBackground'
#         super(RecordBackground, self).initialize(frame)
#         self.tasks[0].traps = self.parent().pattern.traps
#         Record = self.tasks[1]
#         fn0 = Record.dvr.filename
#         Record.fn = fn0.replace(fn0.split('/')[-1], 'background.avi')
#         Record.sigDone.connect(lambda: setattr(Record.dvr, 'filename', fn0))
        
from ..QTask import QTask

class RecordBackground(QTask):
    """"Move particles offscreen, record for nframes, and move particles back"""
    
    def __init__(self, x=None, y=-10., stepSize=0.4, nframes=50, **kwargs):
        super(RecordBackground, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.stepSize = stepSize
        self.nframes = nframes
    
    @property
    def x(self): return self._x
    
    @property
    def y(self): return self._y
    
    @x.setter
    def x(self, x):
        if x is not None: self._y = None
        self._x = x
    
    @y.setter
    def y(self, y):
        if y is not None: self._x = None
        self._y = y
        
    def initialize(self, frame):
        super(RecordBackground, self).initialize(frame)                               
        move1 = self.register('MoveToCoordinate', y=self.y, stepSize=self.stepSize, traps=self.parent().pattern.traps)
        rec = self.register('Record', unblock=False, nframes=self.nframes)
        move2 = self.register('Move', reverse=True)      
        
        self.nframes=0
        f0 = rec.dvr.filename
        rec.fn = f0.replace(f0.split('/')[-1], 'background.avi')
        
        #### Not needed since traps and trajectories are passed through task data (for now...)
        # move1.sigDone.connect(lambda: setattr(move2, 'trajectories', move1.data['trajectories']))
        # move1.sigDone.connect(lambda: setattr(move2, 'traps', move1.data['traps']))

    