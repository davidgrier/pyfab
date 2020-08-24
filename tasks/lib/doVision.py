# -*- coding: utf-8 -*-

from ..QTask import QTask
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)

import ujson as json
import numpy as np
import sys
#sys.path.append('/home/jackie/Desktop')
sys.path.append('/home/group/python')

from pylorenzmie.analysis import Frame

from CNNLorenzMie.Localizer import Localizer
from CNNLorenzMie.Estimator import Estimator
from CNNLorenzMie.filters import no_edges, nodoubles
from CNNLorenzMie.crop_feature import crop_frame, est_crop_frame


#### Converts the input frames to pylorenzmie Frames, sends them out in a signal, and keeps them in a Video

class Filter(object):
    def __init__(self, doNoDoubles=True, doNoEdges=True, doublestol=600, edgetol=200):
        self.doNoDoubles = doNoDoubles
        self.doNoEdges = doNoEdges
        self.doublestol = doublestol
        self.edgetol = edgetol        
    def predict(self, frame):
        if self.doNoDoubles: nodoubles(frame, tol=self.doublestol)
        if self.doNoEdges: no_edges(frame, tol=self.edgetol)
    
class doVision(QTask):
    sigLocalizerChanged = pyqtSignal(object)    
    sigFiltererChanged = pyqtSignal(object)        
    sigEstimatorChanged = pyqtSignal(object)    
    sigRTDone = pyqtSignal(Frame)
    def __init__(self, **kwargs):
        super(doVision, self).__init__(**kwargs)
        self._blocking = False
        self.delay = 10
        
        
        self.setupPredictSettings()
        self.startsetting = 0
        self.rtsetting = self.setting()
        self.ppsetting = 2
        self.rtframes = []
        self.ppframes = []
            
              
    def initialize(self, frame):
        self.localizer = None
        self.filterer = None
        self.estimator = None
        if isinstance(frame, np.ndarray): 
#             frame = None
            print('recieved image frame')
#         self.parent().tasks.source.sigNewFrame.disconnect(self.handleTask)

    
    @pyqtSlot(Frame)
    def handleTask(self, frame):
        if isinstance(frame, np.ndarray): return
        print('{}: {}'.format(self.name, type(frame)))
        
        if self._frame % self.skip != 0:      #### Put all 'skipped' frames from process into a list for post-processing
            print('{}: {}'.format(self.name, type(frame)))        
            self.ppframes.append(frame)     #### This works since _frame=0 before initialization and 0%skip==0
        super(doVision, self).handleTask(frame)
    
    def process(self, frame):
        print('running')
        print(type(frame))
        print(self.startsetting)
        print(self.rtsetting)
        self.predict(frame, self.startsetting, self.rtsetting)
        self.rtframes.append(frame)
        print('running')
        self.sigRTDone.emit(frame)
        
   
    def complete(self):
        self.ppsetting = self.setting(post=True)
        for frame in self.rtframes:
            self.predict(frame, self.rtsetting+1, self.ppsetting)
        for frame in self.ppframes:
            self.predict(frame, self.startsetting, self.ppsetting)
      
    
    def predict(self, frame, start, end):        
        for i in range(start, end+1):
            if i==1: 
                self.localizer.predict(frame);
            if i==2:
                self.filterer.predict(frame)
            if i==3:
                crop_frame(frame)
            if i==4:
                imgs, scales, feats = est_crop_frame(frame)
                self.estimator.predict(imgs, scales, feats)
            if i==5:
                frame.optimize()
            print(frame.to_df())
            
    @property
    def localizer(self):
        return self._localizer
    @localizer.setter
    def localizer(self, localizer):
        self._localizer = localizer or Localizer('tinyholo', weights='_500k')
        self.sigLocalizerChanged.emit(self.localizer)
   
    @property
    def filterer(self):
        return self._filterer
    @filterer.setter
    def filterer(self, filterer):
        self._filterer = filterer or Filter()
        self.sigFiltererChanged.emit(self.filterer)
   
    @property
    def estimator(self):
        return self._estimator
    @estimator.setter
    def estimator(self, estimator):
        if estimator is not None:
            self._estimator = estimator
        else:
#            keras_head_path = '/home/jackie/Desktop/CNNLorenzMie/keras_models/predict_stamp_best'
             keras_head_path = '/home/group/python/CNNLorenzTest/keras_models/predict_stamp_best'
            with open(keras_head_path+'.json', 'r') as f:
                kconfig = json.load(f)
            self.estimator = Estimator(model_path=keras_head_path+'.h5', config_file=kconfig)
        self.sigEstimatorChanged.emit(self.estimator)
    
    
    def setupPredictSettings(self):
        self.rtstate=2
        
        self.rtdetect = True
        self.rtfilt = False
        self.rtcrop = False
        self.rtestimate = False
        self.rtrefine = False
        
        self.ppdetect = True
        self.ppfilt = True
        self.ppcrop = True
        self.ppestimate = False
        self.pprefine = False
    
    def setting(self, post=False):
        if not post:
            if self.rtstate==0: return 9;
            elif self.rtdetect: return 1;
            elif self.rtfilt: return 2;
            elif self.rtcrop: return 3;
            elif self.rtestimate: return 4;
            elif self.rtrefine: return 5;
        else:
            if self.rtstate==1: i=9;
            elif self.ppdetect: i=1;
            elif self.ppfilt: i=2;
            elif self.ppcrop: i=3;
            elif self.ppestimate: i=6;
            elif self.pprefine: i=5;
            return min(self.setting(False), i)
        

        
        
                