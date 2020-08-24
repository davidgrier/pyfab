# -*- coding: utf-8 -*-
# MENU: Start Vision Pipeline

from ..QTask import QTask
from ..QVision3 import QVision 
from PyQt5.QtCore import (pyqtSignal, pyqtSlot)

import json
import sys
#sys.path.append('/home/jackie/Desktop')
sys.path.append('/home/group/python')

from pylorenzmie.analysis import Frame, Video


#### Converts the input frames to pylorenzmie Frames, sends them out in a signal, and keeps them in a Video
class toVision(QTask):
    
    sigNewFrame = pyqtSignal(Frame)
    def __init__(self, nframes=1e6, path=None, **kwargs):
        super(toVision, self).__init__(nframes=nframes, **kwargs)
        self._blocking = False
        self._paused = True
        self.path = path
        self.plmframes = []   
        self.widget = QVision(parent=None, source=self)
       

    def initialize(self, frame):
        frame = None   ## The first frame is from the camera, so ignore it
        self.screen = self.parent().screen
        self.screen.source.sigNewFrame.disconnect(self.handleTask)          ## disconnect from camera and connect to screen (to allow normalization using sample and hold)
        self.screen.sigNewFrame.connect(self.handleTask)                    ## note: pyfab.tasks.source = pyfab.screen.source
        
    def process(self, frame):
        if frame is None: 
#            print('frame {} is None'.format(self._frame))
            return
        plmframe = Frame(framenumber=self._frame, image=frame)
        self.plmframes.append(plmframe)
        self.sigNewFrame.emit(plmframe)
   
    def complete(self):
        self.sigNewFrame.disconnect()
         
    @pyqtSlot(list)
    def writeFrames(self, indices):
        self._busy = True
        for index in indices:
            self.plmframes[i].serialize()
        self._busy = False
        
#         for frame in self.plmframes:
#             print(frame.to_df())
            
#             if self.path is not None:
#                 frame.serialize(save=True, path=path)