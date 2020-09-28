# -*- coding: utf-8 -*-

from ..QTask import QTask
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)
from PyQt5.QtWidgets import QWidget
from .doVisionWidget import Ui_doVisionWidget
from common.QSettingsWidget import QSettingsWidget

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

    sigRealTime = pyqtSignal(Frame)
    sigPost = pyqtSignal(list)
    def __init__(self, **kwargs):
        super(doVision, self).__init__(**kwargs)
        self._blocking = False
        self.source = 'vision'
        
        self.initializeSettings()        
        self.rtframes = []
        self.ppframes = []
        # self.widget = QWidget()
        self.widget = QWidget()         
        self.name = 'doVision'
        self.actual_widget = doVisionWidget(parent=self.parent(), device=self)
    def initialize(self, frame):
        self.localizer = None    #### Call localizer and estimator setters
        self.estimator = None

    
    @pyqtSlot(Frame)
    def handleTask(self, frame):        
        if self._frame % self.skip != 0:      #### Put all 'skipped' frames from process into a list for post-processing
            self.ppframes.append(frame)         #### This works since _frame=0 before initialization and 0%skip==0
        super(doVision, self).handleTask(frame)
    
    def process(self, frame):
#        frame.instrument = self.estimator.instrument
        rt, pp = self.pipelineSettings()
        print("rt={}, pp={}".format(rt, pp))

        self.predict(frame, 0, rt)
        self.rtframes.append(frame)
        self.sigRealTime.emit(frame)

   
    def complete(self):
        rt, pp = self.pipelineSettings()
        # framenumbers = []
        for frame in self.ppframes:
            self.predict(frame, 0, pp)
            # framenumbers.append(frame.framenumber)
        for frame in self.rtframes:
            self.predict(frame, rt, pp)   
            # framenumbers.append(frame.framenumber)
        # self.sigPost(framenumbers)
        self.ppframes.extend(self.rtframes)
        self.sigPost.emit(self.ppframes)
        
    def predict(self, frame, start, end):
        for i in range(start, end):
            if i==0: 
                self.localizer.predict(frame)
                # self.sigBBoxes.emit([bbox for bbox in frame.bboxes if bbox is not None])
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
        
    
class doVisionWidget(QSettingsWidget):
    def __init__(self, parent=None, device=None, **kwargs):
        super(doVisionWidget, self).__init__(parent=parent, device=device, ui=Ui_doVisionWidget(), include=['_paused', 'SRC_paused'], **kwargs)        
        self.tasks = self.parent().tasks      
        self.tasks.sources['realtime'] = self.device.sigRealTime
        self.tasks.sources['post'] = self.device.sigPost
        self.connectUiSignals()
        self.updateUi()

        
    @pyqtSlot()    
    @pyqtSlot(bool)
    def toggleStart(self, running=False):
        self.ui.bstart.setEnabled(not running)
        self.ui.bstop.setEnabled(running)   
        
    @pyqtSlot()
    def start(self):
        self.device._frame = 0
        self.device._busy = False
        print('frame=0')
        # self.devices['SRC'].sigNewFrame.connect(self.device.handleTask)
        self.tasks.queueTask(self.device)
        self.toggleStart(True)
        # print(self.device.__dict__)
           
    @pyqtSlot(int)
    def toggleContinuous(self, state):
        if state==0:
            self.device.nframes = self._store_nframes_setting
            self.ui.nframes.setEnabled(True)
        else:
            self._store_nframes_setting = self.device.nframes
            self.device.nframes = 1e6
            self.ui.nframes.setEnabled(False)
        self.updateUi()
   
    def connectUiSignals(self):       
        #### These three signals should connect to other widgets
        # self.device.sigLocalizerChanged.connect(lambda x: self.setDevice('LOC', x))
        # self.device.sigEstimatorChanged.connect(lambda x: self.setDevice('EST', x))
        # self.device.sigEstimatorChanged.connect(self.devices['SRC'].setInstrument)
        
        self.device.sigDone.connect(self.toggleStart)
        # # self.device.sigBBoxes.connect(self.draw)
        # self.device.sigRealTime.connect(self.redraw)
                
        self.ui.bstart.clicked.connect(self.start)
        self.ui.bstop.clicked.connect(self.device.stop)
 
        
        self.toggleContinuous(2)
        self.ui.continuous.stateChanged.connect(self.toggleContinuous)
        self.ui.continuous.stateChanged.emit(self.ui.continuous.checkState())
        
        
        RTwidgets = [self.ui.rtdetect, self.ui.rtfilt, self.ui.rtcrop, self.ui.rtestimate, self.ui.rtrefine]
        PPwidgets = [self.ui.ppdetect, self.ui.ppfilt, self.ui.ppcrop, self.ui.ppestimate, self.ui.pprefine]
        for i in range(5):
            for j in range(i):
                RTwidgets[i].clicked.connect(lambda _, pp=PPwidgets[j]: pp.setEnabled(False))
            for j in range(4, i-1, -1):
                RTwidgets[i].clicked.connect(lambda _, pp=PPwidgets[j]: pp.setEnabled(True))
                    
#         self.configurePlots()
#         self.configureChildUi()
