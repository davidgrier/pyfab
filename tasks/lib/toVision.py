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
        path = path or self.parent().dvr.filename
        print(path)
        self.video = Video(path=path)
        self._widget = QVision(parent=None, source=self)
   
    def initialize(self, frame):
        frame = None   ## The first frame is from the camera, so ignore it
        self.screen = self.parent().screen
        self.screen.source.sigNewFrame.disconnect(self.handleTask)          ## disconnect from camera and connect to screen (to allow normalization using sample and hold)
        self.screen.sigNewFrame.connect(self.handleTask)                    ## note: pyfab.tasks.source = pyfab.screen.source
        
    def process(self, frame):
        if frame is None: 
            return
        plmframe = Frame(framenumber=self._frame, path=self.video.path, image=frame)
        plmframe = self.video.set_frame(frame=plmframe, framenumber=self._frame)
#        print(plmframe.__dict__)
        self.sigNewFrame.emit(plmframe)
   
    def complete(self):
        self.sigNewFrame.disconnect()


    @pyqtSlot(object)
    def setInstrument(self, instrument):
        self.video.instrument = instrument

    @pyqtSlot()
    def write(self):
        self._busy = True
        self.video.serialize(save=True, omit_feat=['data'])
        self._busy = False         
        
    
    
#    @pyqtSlot(list)
#    def writeFrames(self, indices=None):
#        self._busy = True
#        print('saving to {}'.format(self.path))
#        if indices is None:
#            indices = list(range(len(self.plmframes)))
#        for index in indices:
#            self.plmframes[index].serialize(save=True)
#        self._busy = False
    
#         for frame in self.plmframes:
#             print(frame.to_df())
            
#             if self.path is not None:
#                 frame.serialize(save=True, path=path)