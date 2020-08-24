# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot, pyqtSignal, pyqtProperty, QThread, QObject
#from .QVisionWidget4 import Ui_QVisionWidget
from common.QMultiSettingsWidget import QMultiSettingsWidget
from tasks.lib.doQVisionWidget import Ui_doVisionWidget
from tasks.lib.doVision import doVision as Vision

import sys
#sys.path.append('/home/jackie/Desktop')
sys.path.append('/home/group/python/')

from pylorenzmie.analysis import Video, Frame

import numpy as np
import pyqtgraph as pg

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


"""  QVision manages the GUI for real-time particle tracking. Vision has a source, an estimator+localizer
      -The source is a special task 'toVision' that takes output directly from the screen and converts it to pylorenzmie Frames.
         The source also emits its own sigNewFrame whenever it has a new Frame(), and handles storing+saving Frames/data.
      -QVision also has a main 'device', which is another task 'doVision' which QVision connects to its source to do CNN predictions
      -When the 'device' finishes running, it is disconnected and has _frame set to 0 so it can be reconnected again. Note that doVision 
         doesn't store any frames; it simply performs predictions on any frames it recieves.
      -'toVision' also has a localizer, estimator and filterer. These are all also 'devices' of this widget, so any changes to the QVision UI
         will carry over to the localizer, estimator, etc. 
   
"""
    
class QVision(QMultiSettingsWidget):

    def __init__(self, parent=None, source=None, **kwargs):
        self.ui=Ui_doVisionWidget()
        super(QVision, self).__init__(parent=parent, ui=self.ui, include=['_paused', 'SRC_paused'], **kwargs)        
            
        self.source = source        
        self.setDevice('SRC', self.source)
#        self.tasks = source.parent().tasks
        self.screen = source.parent().screen
        
        self.device = Vision(nframes=100)
        self.device.name = 'doVision'
        self.device.sigLocalizerChanged.connect(lambda x: self.setDevice('LOC', x))
        self.device.sigFiltererChanged.connect(lambda x: self.setDevice('FILT', x))
        self.device.sigEstimatorChanged.connect(lambda x: self.setDevice('EST', x))
        

        self.connectUiSignals()
        self.updateUi()
        
        self.rois = None
        self.pen = pg.mkPen(color='b', width=5)
    
            
    @pyqtSlot()
    def start(self):
        if self.source is None:
            return
        else:
            self.device._frame = 0
        self.source.sigNewFrame.connect(self.device.handleTask)
        self.device.sigDone.connect(self.stop)
        self.device.sigRTDone.connect(lambda: self.draw)
        print('started')
        print(self.device.__dict__)
        self.ui.bstart.setEnabled(False)
        self.ui.bstop.setEnabled(True)       
     
    @pyqtSlot()
    def stop(self):
#        sleep()
        self.source.sigNewFrame.disconnect(self.device.handleTask)
        self.device.stop()
        self.ui.bstart.setEnabled(True)
        self.ui.bstop.setEnabled(False)

    @pyqtSlot()
    def toggleRealTime(self):
        check = self.ui.rtstate
        if check.checkState() == 0:
            check.setText('Disabled')
        elif check.checkState() == 1:
            check.setText('Continuous')
            self.device.nframes = self.source.nframes
            self.ui.nframes.setEnabled(False)
        elif check.checkState() == 2:
            check.setText('Enabled')
            self.ui.nframes.setEnabled(True)

    def connectUiSignals(self):
        self.ui.rtstate.clicked.connect(self.toggleRealTime)
        self.ui.bstart.clicked.connect(self.start)
        self.ui.bstop.clicked.connect(self.stop)
        
        
        RTwidgets = [self.ui.rtdetect, self.ui.rtfilter, self.ui.rtcrop, self.ui.rtestimate, self.ui.rtrefine]
        PPwidgets = [self.ui.ppdetect, self.ui.ppfilter, self.ui.ppcrop, self.ui.ppestimate, self.ui.pprefine]
        for i in range(5):
            for j in range(i):
                RTwidgets[i].clicked.connect(lambda _, pp=PPwidgets[j]: pp.setEnabled(False))
            for j in range(4, i-1, -1):
                RTwidgets[i].clicked.connect(lambda _, pp=PPwidgets[j]: pp.setEnabled(True))
                    
#         self.configurePlots()
#         self.configureChildUi()
    
    def draw(self, plmframe):
        rois = []
        for bbox in plmframe.bboxes:
            if bbox is not None:
                x, y, w, h = bbox
                roi = pg.RectROI([x-w//2, y-h//2], [w, h], pen=self.pen)
                self.source.screen.addOverlay(roi)
                rois.append(roi)
        print(rois)
        return rois













#     #
#     # Configuration for QHVM-specific functionality
#     #
#     def configurePlots(self):
#         self.ui.plot1.setBackground('w')
#         self.ui.plot1.getAxis('bottom').setPen(0.1)
#         self.ui.plot1.getAxis('left').setPen(0.1)
#         self.ui.plot1.showGrid(x=True, y=True)
#         self.ui.plot1.setLabel('bottom', 'a_p [um]')
#         self.ui.plot1.setLabel('left', 'n_p')
#         self.ui.plot1.addItem(pg.InfiniteLine(self.instrument.n_m, angle=0,
#                                               pen=pg.mkPen('k', width=3, style=Qt.DashLine)))
#         self.ui.plot2.setBackground('w')
#         self.ui.plot2.getAxis('bottom').setPen(0.1)
#         self.ui.plot2.getAxis('left').setPen(0.1)
#         self.ui.plot2.showGrid(x=True, y=True)
#         self.ui.plot2.setLabel('bottom', 't (s)')
#         self.ui.plot2.setLabel('left', 'z(t)')

#     def configureChildUi(self):
#         self.ui.bDetect.setEnabled(True)
#         self.ui.bEstimate.setEnabled(True)
#         self.ui.bRefine.setEnabled(True)

#     #
#     # Overwrite QVision methods for HVM
#     #
#     @pyqtSlot()
#     def plot(self):
#         if self.estimate:
#             a_p = []
#             n_p = []
#             framenumbers = []
#             trajectories = []
#             for frame in self.video.frames:
#                 for feature in frame.features:
#                     a_p.append(feature.model.particle.a_p)
#                     n_p.append(feature.model.particle.n_p)
#             for trajectory in self.video.trajectories:
#                 z_p = []
#                 for feature in trajectory.features:
#                     z_p.append(feature.model.particle.z_p)
#                 trajectories.append(z_p)
#                 framenumbers.append(trajectory.framenumbers)
#             # Characterization plot
#             data = np.vstack([a_p, n_p])
#             if len(a_p) > 0:
#                 pdf = gaussian_kde(data)(data)
#                 norm = pdf/pdf.max()
#                 rgbs = []
#                 for val in norm:
#                     rgbs.append(self.getRgb(cm.cool, val))
#                 pos = [{'pos': data[:, i],
#                         'pen': pg.mkPen(rgbs[i], width=2)} for i in range(len(a_p))]
#                 scatter = pg.ScatterPlotItem()
#                 self.ui.plot1.addItem(scatter)
#                 scatter.setData(pos)
#                 # z(t) plot
#                 grayscale = np.linspace(0, 1, len(trajectories), endpoint=True)
#                 for j in range(len(trajectories)):
#                     z_p = trajectories[j]
#                     f = np.array(framenumbers[j])
#                     curve = pg.PlotCurveItem()
#                     self.ui.plot2.addItem(curve)
#                     curve.setPen(pg.mkPen(self.getRgb(cm.gist_rainbow,
#                                                       grayscale[j]), width=2))
#                     curve.setData(x=f/self.video.fps, y=z_p)
#         self.sigCleanup.emit()
