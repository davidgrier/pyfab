# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, QObject
from .QVisionWidget import Ui_QVisionWidget
from common.QSettingsWidget import QSettingsWidget

from pylorenzmie.analysis import Video, Frame

import numpy as np
import pyqtgraph as pg
import json

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)


class QWriter(QObject):

    finished = pyqtSignal()

    def __init__(self, data, filename):
        super(QWriter, self).__init__()
        self.data = data
        self.filename = filename

    @pyqtSlot()
    def write(self):
        logger.info('Saving...')
        data = json.dumps(self.data)
        with open(self.filename, 'w') as f:
            f.write(data+'\n')
        logger.info('{} saved!'.format(self.filename))
        self.finished.emit()


class QVision(QSettingsWidget):

    sigPlot = pyqtSignal()
    sigCleanup = pyqtSignal()
    sigPost = pyqtSignal()

    def __init__(self, parent=None):

        # Set serialized properties
        self.nskip = 0
        self.linkTol = 20.
        self.threshold = 50.
        self.counter = self.nskip
        self._realTime = True
        self._postProcess = False
        self.saveFrames = False
        self.saveTrajectories = False
        self.saveFeatureData = False

        # Set non-serialized properties
        self.detect = False
        self.estimate = False
        self.refine = False
        self.localizer = None
        self.estimator = None

        # Setup Ui
        self.ui = Ui_QVisionWidget()
        super(QVision, self).__init__(parent=parent, device=self, ui=self.ui)
        self.connectVisionSignals()

        self.jansen = None

        # pylorenzmie objects
        self.instrument = None
        self.video = Video(instrument=self.instrument)

        # Miscellaneous
        self.filename = None
        self.frames = []
        self.framenumbers = []

        self._thread = None

        self.recording = False

        self.rois = None
        self.pen = pg.mkPen(color='b', width=5)

    #
    # Methods to overwrite in subclass
    #
    def configurePlots(self):
        pass

    def init_pipeline(self):
        pass

    def clear_pipeline(self):
        pass

    def draw(self, detections):
        return []

    #
    # Ui handling
    #
    def connectVisionSignals(self):
        self.ui.bDetect.clicked.connect(self.handleDetect)
        self.ui.bEstimate.clicked.connect(self.handleEstimate)
        self.ui.bRefine.clicked.connect(self.handleRefine)
        self.sigCleanup.connect(self.cleanup)
        self.sigPlot.connect(self.plot)
        self.sigPost.connect(self.post_process)

    def configureUi(self):
        self.ui.bDetect.setChecked(self.detect)
        self.ui.bEstimate.setChecked(self.estimate)
        self.ui.bRefine.setChecked(self.refine)
        self.ui.bDetect.setEnabled(False)
        self.ui.bEstimate.setEnabled(False)
        self.ui.bRefine.setEnabled(False)

    def closeEvent(self):
        logger.debug('Closing vision pipeline')
        self.remove()
        self.clear_pipeline()

    #
    # Special getters and setters
    #
    @property
    def realTime(self):
        return self._realTime

    @realTime.setter
    def realTime(self, realTime):
        self._postProcess = not realTime
        self._realTime = realTime

    @property
    def postProcess(self):
        return self._postProcess

    @postProcess.setter
    def postProcess(self, postProcess):
        self._postProcess = postProcess
        self._realTime = not postProcess
        if postProcess:
            self.remove()
            self.rois = None

    #
    # Slots
    #
    @pyqtSlot()
    def plot(self):
        self.sigCleanup.emit()

    @pyqtSlot(np.ndarray)
    def process(self, image):
        self.remove()
        if self.counter == 0:
            self.counter = self.nskip
            i = self.jansen.dvr.framenumber
            if self.realTime:
                frames, detections = self.predict([image], [i])

                frame = frames[0]
                self.rois = self.draw(detections[0])
                if self.jansen.dvr.is_recording():
                    self.recording = True
                    if len(frame.features) != 0:
                        self.video.add(frames)
            else:
                if self.jansen.dvr.is_recording():
                    self.recording = True
                    self.frames.append(image)
                    self.framenumbers.append(i)
        else:
            self.counter -= 1
            self.rois = None
        if self.recording:
            item1, item2 = (self.ui.plot1.getPlotItem(), self.ui.plot2.getPlotItem())
            plots1, plots2 = (item1.listDataItems(), item2.listDataItems())
            for plot in plots1:
                item1.removeItem(plot)
            for plot in plots2:
                item2.removeItem(plot)
            if not self.jansen.dvr.is_recording():
                self.recording = False
                self.sigPost.emit()
            

    @pyqtSlot()
    def cleanup(self):
        if self.saveFrames or self.saveTrajectories:
            omit, omit_feat = ([], [])
            if not self.saveFrames:
                omit.append('frames')
            if not self.saveTrajectories:
                omit.append('trajectories')
            if not self.saveFeatureData:
                omit_feat.append('data')
            filename = self.jansen.dvr.filename.split(".")[0] + '.json'
            out = self.video.serialize(omit=omit,
                                       omit_frame=['data'],
                                       omit_feat=omit_feat)
            self._writer = QWriter(out, filename)
            self._thread = QThread()
            self._writer.moveToThread(self._thread)
            self._thread.started.connect(self._writer.write)
            self._writer.finished.connect(self.close)
            self._thread.start()
        self.video = Video(instrument=self.instrument)

    @pyqtSlot()
    def close(self):
        logger.debug('Shutting down save thread')
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        self._video = None
        logger.debug('Save thread closed')

    @pyqtSlot(bool)
    def handleDetect(self, selected):
        if selected:
            self.detect = True
            self.estimate = False
            self.refine = False
            self.ui.bEstimate.setChecked(False)
            self.ui.bRefine.setChecked(False)
            self.init_pipeline()
        else:
            self.clear_pipeline()

    @pyqtSlot(bool)
    def handleEstimate(self, selected):
        if selected:
            self.detect = True
            self.estimate = True
            self.refine = False
            self.ui.bDetect.setChecked(False)
            self.ui.bRefine.setChecked(False)
            self.init_pipeline()
        else:
            self.clear_pipeline()

    @pyqtSlot(bool)
    def handleRefine(self, selected):
        if selected:
            self.detect = True
            self.estimate = True
            self.refine = True
            self.ui.bDetect.setChecked(False)
            self.ui.bEstimate.setChecked(False)
            self.init_pipeline()
        else:
            self.clear_pipeline()

    @pyqtSlot()
    def post_process(self):
        self.video.fps = self.jansen.screen.fps
        if self.detect:
            if not self.realTime:
                self.jansen.screen.source.blockSignals(True)
                self.jansen.screen.pauseSignals(True)
                frames, detections = self.predict(self.frames,
                                                  self.framenumbers,
                                                  post=True)
                self.video.add(frames)
                self.jansen.screen.source.blockSignals(False)
                self.jansen.screen.pauseSignals(False)
            self.video.set_trajectories(search_range=self.linkTol,
                                        memory=int(self.nskip+3))
        self.frames = []
        self.framenumbers = []
        self.sigPlot.emit()

    #
    # Methods
    #
    def predict(self, images, framenumbers, post=False):
        frames = []
        detections = []
        for image in images:
            frames.append(Frame())
            detections.append([])
        return frames, detections

    def remove(self):
        rois = self.rois
        if rois is not None:
            for rect in rois:
                self.jansen.screen.removeOverlay(rect)

    def getRgb(self, cmap, gray):
        clr = cmap(gray)
        rgb = []
        for i in range(len(clr)):
            rgb.append(int(255*clr[i]))
        return tuple(rgb)
