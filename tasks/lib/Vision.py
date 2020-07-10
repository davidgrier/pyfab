# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot, pyqtSignal, pyqtProperty, QThread, QObject
from .QTask import QTask

from pylorenzmie.analysis import Video, Frame
from pylorenzmie.theory import Instrument

import CNNLorenzMie as cnn

import numpy as np
#import pyqtgraph as pg
import ujson as json
# from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import os
# import json
# from time import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

class settings(Enum):
    raw = 0
    detect = 1
    estimate = 2
    refine = 3
    
class Vision(QTask):
    
#     sigDone = pyqtSignal(self)    ## Should tasks always return themselves? Maybe add this to QTask.py
    sigNewVisionFrame = pyqtSignal(Frame)
    sigNewVisionVideo = pyqtSignal(Video)

    def __init__(self, **kwargs):
        super(Vision, self).__init__(blocking=False, **kwargs)
        """Perform initialization operations"""
        logger.debug('Initializing')
         # Initialize serialized properties
        self._linkTol = 20.
        self._confidence = 50.
        self._maxSize = 603
#         self._realTime = True
#         self._postProcess = False
#         self._saveFrames = False
#         self._saveTrajectories = False
#         self._saveFeatureData = False


        # Set non-serialized properties
        self.setting = settings.raw
        self.realtime = settings.detect
        self.post = settings.estimate
        self.link = settings.detect
        self.localizer = None
        self.estimator = None

        # pylorenzmie objects
        self.instrument = None
        self.video = Video(instrument=self.instrument)

        self.filename = None
        self.frames = []
        self.framenumbers = []
        
        self.count = 0
        self.runTime = self.nframes
    
    
        
    #
    # Slots
    #
    def resetVideo():
        self.video._frames = []
        self.video._trajectories = []
        self.count = 0
        
    def initialize(self, frame):
        self.resetVideo()
        
    def process(self, frame):
        if self.count is self.runTime:
            self.setting = self.realtime
            self.video.frames = self.predict(self.video.frames, post=True)
        elif self.count < self.runTime:
            self.video.add([ self.predict([frame]) ])
        self.count += 1
#         self.remove()
#         if self.counter == 0:
#             self.counter = self._nskip
#             i = self.jansen.dvr.framenumber
#             if self.realTime:
#                 frames, detections = self.predict([image], [i])
#                 frame = frames[0]
#                 self.sigNewFrame.emit(frame)

#                 if self.jansen.dvr.is_recording():
#                     self.recording = True
#                     if len(frame.features) != 0:
#                         self.video.add(frames)

 #####  todo #####
    def complete(self):
#         self.video.fps = self.jansen.screen.fps   ## I'm not sure how to handle this line yet
        if not self.realTime:    
            vframes, detections = self.predict(self.frames, self.framenumbers)
            self.video.add(vframes)
        self.video.set_trajectories(verbose=False,
                                        search_range=self.linkTol,
                                        memory=int(self.skip+3))
        self.sigNewVisionVideo(self.video)
#     @pyqtSlot()
#     def cleanup(self):
#         if self.saveFrames or self.saveTrajectories:
#             omit, omit_feat = ([], [])
#             if not self.saveFrames:
#                 omit.append('frames')
#             if not self.saveTrajectories:
#                 omit.append('trajectories')
#             if not self.saveFeatureData:
#                 omit_feat.append('data')
#             filename = self.jansen.dvr.filename.split(".")[0] + '.json'
#             out = self.video.serialize(omit=omit,
#                                        omit_frame=['data'],
#                                        omit_feat=omit_feat)
#             self._writer = QWriter(out, filename)
#             self.jansen.screen.sigNewFrame.connect(self._writer.write)
#             self._thread = QThread()
#             self._writer.moveToThread(self._thread)
#             self._thread.started.connect(self._writer.start)
#             self._writer.finished.connect(self.close)
#             self._thread.start()
#         self.video = Video(instrument=self.instrument)

#     @pyqtSlot()
#     def close(self):
#         logger.debug('Shutting down save thread')
#         self._thread.quit()
#         self._thread.wait()
#         self._thread = None
#         self._video = None
#         logger.debug('Save thread closed')     

 #################


    
    def predict(self, frames, post=False):
        finished = self.post if post else self.realtime
        if self.setting is settings.raw:
           

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
