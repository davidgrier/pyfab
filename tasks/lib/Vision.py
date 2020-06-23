# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot, pyqtSignal, pyqtProperty, QThread, QObject
from ..QTask import QTask

from pylorenzmie.analysis import Video, Frame

import numpy as np
import pyqtgraph as pg
import ujson as json

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

from pylorenzmie.analysis import Frame
from pylorenzmie.theory import Instrument

from scipy.stats import gaussian_kde

import CNNLorenzMie as cnn
import numpy as np
import pyqtgraph as pg
import matplotlib.cm as cm
import os
import json
from time import time
class Vision(QTask):
    
#     sigDone = pyqtSignal(self)    ## Should tasks always return themselves? Maybe add this to QTask.py
    sigNewVisionFrame = pyqtSignal(Frame)
    sigNewVisionVideo = pyqtSignal(Video)

    def __init__(self, **kwargs):
        super(Vision, self).__init__(blocking=False, **kwargs)
       
        
    def initialize(self, frame):
        """Perform initialization operations"""
        logger.debug('Initializing')
         # Initialize serialized properties
        self._linkTol = 20.
        self._confidence = 50.
        self._maxSize = 603
        self._realTime = True
#         self._postProcess = False

        # Set non-serialized properties
        self.detect = False
        self.estimate = False
        self.refine = False
        self.localizer = None
        self.estimator = None

        # pylorenzmie objects
        self.instrument = None
        self.video = Video(instrument=self.instrument)

        self.filename = None
        self.frames = []
        self.framenumbers = []
                    
    def complete(self):
#         self.video.fps = self.jansen.screen.fps   ## I'm not sure how to handle this line yet
        if not self.realTime:    
            vframes, detections = self.predict(self.frames, self.framenumbers)
            self.video.add(vframes)
        self.video.set_trajectories(verbose=False,
                                        search_range=self.linkTol,
                                        memory=int(self.skip+3))
        self.sigNewVisionVideo(self.video)


    

    #
    # Methods
    #
    def predict(self, images, framenumbers, post=False):   #### This is a placeholder method for CNNLorenzMie framework
        frames = []
        detections = []
        for image in images:
            frames.append(Frame())
            detections.append([])
        return frames, detections


    def getRgb(self, cmap, gray):
        clr = cmap(gray)
        rgb = []
        for i in range(len(clr)):
            rgb.append(int(255*clr[i]))
        return tuple(rgb)

    #
    # PyQt serialized properties
    #
    @pyqtProperty(bool)
    def realTime(self):
        return self._realTime

    @realTime.setter
    def realTime(self, realTime):
        self._postProcess = not realTime
        self._realTime = realTime

#     @pyqtProperty(bool)
#     def postProcess(self):
#         return self._postProcess

#     @postProcess.setter
#     def postProcess(self, postProcess):
#         self._postProcess = postProcess
#         self._realTime = not postProcess


    @pyqtProperty(float)
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, thresh):
        self._confidence = thresh

    @pyqtProperty(int)
    def maxSize(self):
        return self._maxSize

    @maxSize.setter
    def maxSize(self, l):
        m = max(self.jansen.screen.source.width,
                self.jansen.screen.source.height)
        self.ui.maxSize.setMaximum(m)
        self._maxSize = l

    @pyqtProperty(float)
    def linkTol(self):
        return self._linkTol

    @linkTol.setter
    def linkTol(self, tol):
        self._linkTol = tol
