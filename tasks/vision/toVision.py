# -*- coding: utf-8 -*-
# MENU: Start Vision Pipeline

from ..QTask import QTask
from .toVisionWidget import Ui_toVisionWidget
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QAbstractTableModel, QAbstractListModel, Qt)
from common.QSettingsWidget import QSettingsWidget

import json
import sys
sys.path.append('/home/jackie/Desktop')
# sys.path.append('/home/group/python')

from pylorenzmie.analysis import Frame, Video

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)

#### Converts the input frames to pylorenzmie Frames, sends them out in a signal, and keeps them in a Video
class toVision(QTask):
    
    sigNewFrame = pyqtSignal(Frame)
    def __init__(self, nframes=1e6, path=None, **kwargs):
        super(toVision, self).__init__(nframes=nframes, **kwargs)
        self.source = 'screen'
        self._blocking = False
        self._paused = True
        
        path = path or self.parent().dvr.filename

        self.video = Video(path=path)
        self.widget = toVisionWidget(parent=self.parent(), device=self)
        
    def process(self, frame):
        if frame is None: 
            return
        plmframe = Frame(framenumber=self._frame, path=self.video.path, image=frame)
        plmframe = self.video.set_frame(frame=plmframe, framenumber=self._frame)
        self.widget.ui.FramesView.model().layoutChanged.emit()
        self.sigNewFrame.emit(plmframe)
   
    def complete(self):
        self.sigNewFrame.disconnect()

    @pyqtSlot(object)
    def setInstrument(self, instrument):
        self.video.instrument = instrument

    @pyqtSlot()
    def write(self):
        self._busy = True
        self.video.serialize(save=True, omit_feat=['data'], framenumbers=self.widget.ui.SelectedFramesView.model().framenumbers)
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
        
        
    

"""  QVision manages the GUI for real-time particle tracking. Vision has a source, an estimator+localizer
      -The source is a special task 'toVision' that takes output directly from the screen and converts it to pylorenzmie Frames.
         The source also emits its own sigNewFrame whenever it has a new Frame(), and handles storing+saving Frames/data.
      -QVision also has a main 'device', which is another task 'doVision' which QVision connects to its source to do CNN predictions
      -When the 'device' finishes running, it is disconnected and has _frame set to 0 so it can be reconnected again. Note that doVision 
         doesn't store any frames; it simply performs predictions on any frames it recieves.
      -'toVision' also has a localizer, estimator and filterer. These are all also 'devices' of this widget, so any changes to the QVision UI
         will carry over to the localizer, estimator, etc. 
"""
    
class toVisionWidget(QSettingsWidget):    

    def __init__(self, parent=None, device=None, **kwargs):
        device.recordFrames = lambda: self.start()
        device.stopRecordFrames = lambda: self.stop()
        super(toVisionWidget, self).__init__(parent=parent, device=device, ui=Ui_toVisionWidget(), **kwargs)        
                        
        self.ui.FramesView.setModel( Model1(video=device.video) )
        self.ui.FramesView.setSelectionMode(3)
        self.ui.SelectedFramesView.setModel( Model2(video=device.video) ) 
        
        self.connectUiSignals()
        self.updateUi()
        
        

    @pyqtSlot()
    def start(self): self.device.pause(False)
    
    @pyqtSlot()
    def stop(self): self.device.pause(True)
    
    @pyqtSlot(int)
    def toggleContinuous(self, state):
        if state==0:
            self.device.nframes = self._store_nframes_setting
            self.ui.stopFrameRecordSettings.setCurrentIndex(0)
            # self.ui.nframes.setEnabled(True)
        else:
            self._store_nframes_setting = self.device.nframes
            self.device.nframes = 1e6
            self.ui.stopFrameRecordSettings.setCurrentIndex(1)
            # self.ui.nframes.setEnabled(False)
        self.updateUi()   
        
    @pyqtSlot(int)
    def toggleStartWithDVR(self, state):
        if state==0:
            self._store_delay_setting = self.device.delay
            self.device.delay = 0
            self.device.parent().dvr.recordButton.clicked.disconnect(self.start)
            self.ui.startFrameRecordSettings.setCurrentIndex(0)
        else:
            self.device.delay = self._store_delay_setting
            self.device.parent().dvr.recordButton.clicked.connect(self.start)
            self.ui.startFrameRecordSettings.setCurrentIndex(1)
        self.updateUi()

    @pyqtSlot(int)
    def toggleStopWithDVR(self, state):
        if state==0:
            self.device.parent().dvr.stopButton.clicked.disconnect(self.stop)
            self.ui.stopFrameRecordSettings.setEnabled(True)
        else:
            self.device.parent().dvr.stopButton.clicked.connect(self.stop)
            self.ui.stopFrameRecordSettings.setEnabled(False)
        self.updateUi()



    def connectUiSignals(self):       
        self._store_delay_setting = self.device.delay
        self._store_nframes_setting = self.device.nframes
        self.ui.startWithDVR.stateChanged.connect(self.toggleStartWithDVR)
        self.ui.stopWithDVR.stateChanged.connect(self.toggleStopWithDVR)
        self.ui.continuous.stateChanged.connect(self.toggleContinuous)


        view = self.ui.FramesView
        model = self.ui.SelectedFramesView.model()
        view.clicked.connect(lambda: model.setFramenumbers( [model.video.framenumbers[index.row()] for index in view.selectedIndexes()]))
 
        self.ui.bsave.clicked.connect(self.device.write)
        
        
        

class Model1(QAbstractListModel):
    def __init__(self, video=None, **kwargs):
        super(Model1, self).__init__(**kwargs)
        self.video = video
        
    def data(self, index, role):
        if role == Qt.DisplayRole:
            frame = self.video.frames[index.row()]
            return 'Frame {}   |   {} Features'.format(frame.framenumber, len(frame.bboxes))
            
    def rowCount(self, index):
        return len(self.video.frames)    
    
    
    
class Model2(QAbstractTableModel):
    def __init__(self, video=None, framenumbers=[], **kwargs):
        super(Model2, self).__init__(**kwargs)
        self.video = video
        self.setFramenumbers(framenumbers)
        self.columns = ['frame', 'bbox', 'r_p', 'a_p', 'n_p']
    
    @property 
    def framenumbers(self):
        return self._framenumbers
    
    @pyqtSlot(list)
    def setFramenumbers(self, fnums):
        self._framenumbers = fnums
        self.indices = []
        for i, frame in enumerate(self.video.get_frames(fnums)):
            for j in range(len(frame.features)):
                self.indices.append((i, j))
        self.layoutChanged.emit()
    
    def data(self, index, role):
        if role == Qt.DisplayRole:
            i, j = self.indices[index.row()]
            frame = self.video.get_frame(self.framenumbers[i])
            col = index.column()
            if col==0: 
                return frame.framenumber
            elif col==1: 
                return str(frame.bboxes[j])
            else:
                feature = frame.features[j]
            if feature.model is None: 
                return 'N/A'
            elif col==2: 
                return '({}, {}, {})'.format(feature.model.particle.x_p, feature.model.particle.y_p, feature.model.particle.z_p)
            elif col==3:
                return feature.model.particle.a_p
            elif col==4:
                return feature.model.particle.n_p
            
    def headerData(self, section, orientation, role):
    # section is the index of the column/row.
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.columns[section]
        
            # if orientation == Qt.Vertical:
            #     return str(self._data.index[section])
    def rowCount(self, index):
        return len(self.indices)
    
    def columnCount(self, index):
        return 5


 
