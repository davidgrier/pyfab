# -*- coding: utf-8 -*-

from ..QTask import QTask
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)

import ujson as json
import numpy as np
import sys
sys.path.append('/home/jackie/Desktop')
#sys.path.append('/home/group/python')

from pylorenzmie.analysis import Frame

from CNNLorenzMie.Localizer import Localizer
from CNNLorenzMie.Estimator import Estimator
from CNNLorenzMie.filters import no_edges, nodoubles
from CNNLorenzMie.crop_feature import crop_frame, est_crop_frame


#### Converts the input frames to pylorenzmie Frames, sends them out in a signal, and keeps them in a Video
    
class doVision(QTask):
    
    keras_head_path = '/home/jackie/Desktop/CNNLorenzMie/keras_models/predict_stamp_best'
    # keras_head_path = '/home/group/python/CNNLorenzTest/keras_models/predict_stamp_best'
            
    sigLocalizerChanged = pyqtSignal(object)    
    sigFiltererChanged = pyqtSignal(object)        
    sigEstimatorChanged = pyqtSignal(object)    
    sigBBoxes = pyqtSignal(list)
    def __init__(self, **kwargs):
        super(doVision, self).__init__(**kwargs)
        self._blocking = False
        # self.delay = 10
        
        self.initializeSettings()        
        self.rtframes = []
        self.ppframes = []
            
              
    def initialize(self, frame):
        self.localizer = None    #### Call localizer and estimator setters
        self.estimator = None
#         self.parent().tasks.source.sigNewFrame.disconnect(self.handleTask)

    
    @pyqtSlot(Frame)
    def handleTask(self, frame):        
        if self._frame % self.skip != 0:      #### Put all 'skipped' frames from process into a list for post-processing
            self.ppframes.append(frame)         #### This works since _frame=0 before initialization and 0%skip==0
#            self.ppframes.append(self._frame)    
        super(doVision, self).handleTask(frame)
    
    def process(self, frame):
#        frame.instrument = self.estimator.instrument
        rt, pp = self.pipelineSettings()
        print("rt={}, pp={}".format(rt, pp))
        # print(list(range(0, rt)))
        # print(list(range(rt, pp)))
        # print(list(range(0, pp)))
        self.predict(frame, 0, rt)
        self.rtframes.append(frame)
#        self.rtframes.append(self._frame)
#        print(frame.__dict__)

   
    def complete(self):
        rt, pp = self.pipelineSettings()
        for frame in self.rtframes:
            self.predict(frame, rt, pp)   
        for frame in self.ppframes:
            self.predict(frame, 0, pp)
      
    
    def predict(self, frame, start, end):
        for i in range(start, end):
            if i==0: 
                self.localizer.predict(frame)
                self.sigBBoxes.emit([bbox for bbox in frame.bboxes if bbox is not None])
                continue
            if i==1:
                self.filter(frame); continue;
            if i==2:
                crop_frame(frame); continue;
            if i==3:
                imgs, scales, feats = est_crop_frame(frame)
                self.estimator.predict(imgs, scales, feats)
                continue
            if i==4:
                frame.optimize(); continue;
        print(frame.to_df())    
            
    def filter(self, frame):
        if self.doNoDoubles: nodoubles(frame, tol=self.doublestol)
        if self.doNoEdges: no_edges(frame, tol=self.edgetol)
    @property
    def localizer(self):
        return self._localizer
    @localizer.setter
    def localizer(self, localizer):
        self._localizer = localizer or Localizer('tinyholo', weights='_500k')
        self.sigLocalizerChanged.emit(self.localizer)
   
    @property
    def estimator(self):
        return self._estimator
    @estimator.setter
    def estimator(self, estimator):
        if estimator is not None:
            self._estimator = estimator
        else:
            with open(self.keras_head_path+'.json', 'r') as f:
                kconfig = json.load(f)
            self.estimator = Estimator(model_path=self.keras_head_path+'.h5', config_file=kconfig)
        self.sigEstimatorChanged.emit(self.estimator)
    
    
    def initializeSettings(self):        
        #### Prediction pipeline settings
        self.rtenabled = True
        self.rtdetect = True
        self.rtfilt = False
        self.rtcrop = False
        self.rtestimate = False
        self.rtrefine = False
        
        self.ppenabled = False
        self.ppdetect = True
        self.ppfilt = True
        self.ppcrop = True
        self.ppestimate = False
        self.pprefine = False
        
        #### Filter settings 
        self.doNoDoubles = True
        self.doNoEdges = True
        self.doublestol = 600
        self.edgetol = 200        
        
    
    def pipelineSettings(self):
        if not self.rtenabled: rt=0;
        elif self.rtdetect: rt=1;
        elif self.rtfilt: rt=2;
        elif self.rtcrop: rt=3;
        elif self.rtestimate: rt=4;
        elif self.rtrefine: rt=5;
        else: rt=0;                #### If nothing is selected, don't do anything 
        
        if not self.ppenabled: pp=0;
        elif self.ppdetect: pp=1;
        elif self.ppfilt: pp=2;
        elif self.ppcrop: pp=3;
        elif self.ppestimate: pp=4;
        elif self.pprefine: pp=5;
        else: pp=0;

        return rt, max(rt, pp)    #### If rt>pp, use realtime setting instead. (note: condsider removing this)
        

        
        
                