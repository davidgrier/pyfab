# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot, pyqtSignal, QThread, QObject
from .QVisionWidget import Ui_QVisionWidget

from pylorenzmie.processing import Video, Frame

import numpy as np
import pyqtgraph as pg
import json

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.setLevel(logging.DEBUG)


class QWriter(QObject):

    finished = pyqtSignal()

    def __init__(self, data, filename):
        super(QWriter, self).__init__()
        self.data = json.dumps(data)
        self.filename = filename

    @pyqtSlot()
    def write(self):
        logger.info('Saving...')
        data = self.data
        n = len(data)
        with open(self.filename, 'w') as f:
            j = 0
            while j < n:
                f.write(data[j])
            f.write('\n')
        logger.info('{} saved!'.format(self.filename))
        self.finished.emit()


class QVision(QWidget):

    sigPlot = pyqtSignal()
    sigCleanup = pyqtSignal()

    def __init__(self, parent=None):
        super(QVision, self).__init__(parent)
        self.ui = Ui_QVisionWidget()
        self.ui.setupUi(self)

        self.jansen = None

        self.instrument = None
        self.video = Video(instrument=self.instrument)

        self.filename = None
        self.frames = []
        self.framenumbers = []

        self._thread = None

        self.recording = False

        self.detect = False
        self.estimate = False
        self.refine = False
        self.nskip = 0
        self.link_tol = 20.
        self.counter = self.nskip
        self.real_time = True
        self.save_frames = False
        self.save_trajectories = False
        self.save_feature_data = False  # TODO: config

        self.rois = None
        self.pen = pg.mkPen(color='b', width=5)

        self.configureUi()
        self.connectSignals()

    def connectSignals(self):
        self.ui.breal.toggled.connect(self.handleRealTime)
        self.ui.bpost.toggled.connect(self.handlePost)
        self.ui.checkFrames.clicked.connect(self.handleSaveFrames)
        self.ui.checkTrajectories.clicked.connect(self.handleSaveTrajectories)
        self.ui.checkFeatureData.clicked.connect(self.handleSaveFeatureData)
        self.ui.bDetect.clicked.connect(self.handleDetect)
        self.ui.bEstimate.clicked.connect(self.handleEstimate)
        self.ui.bRefine.clicked.connect(self.handleRefine)
        self.ui.skipBox.valueChanged.connect(self.handleSkip)
        self.ui.spinTol.valueChanged.connect(self.handleLink)
        self.sigCleanup.connect(self.cleanup)
        self.sigPlot.connect(self.plot)

    def configureUi(self):
        self.ui.bDetect.setChecked(self.detect)
        self.ui.bEstimate.setChecked(self.estimate)
        self.ui.bRefine.setChecked(self.refine)
        self.ui.bDetect.setEnabled(False)
        self.ui.bEstimate.setEnabled(False)
        self.ui.bRefine.setEnabled(False)
        self.ui.checkFrames.setChecked(self.save_frames)
        self.ui.checkTrajectories.setChecked(self.save_trajectories)
        self.ui.breal.setChecked(self.real_time)
        self.ui.skipBox.setProperty("value", self.nskip)
        self.ui.spinTol.setProperty("value", self.link_tol)

    def configurePlots(self):
        pass

    @pyqtSlot()
    def plot(self):
        self.sigCleanup.emit()

    @pyqtSlot(np.ndarray)
    def process(self, image):
        self.remove(self.rois)
        if self.counter == 0:
            self.counter = self.nskip
            i = self.jansen.dvr.framenumber
            if self.real_time:
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
        if self.recording and not self.jansen.dvr.is_recording():
            self.recording = False
            if not self.real_time:
                self.post_process()
            self.sigPlot.emit()

    @pyqtSlot()
    def cleanup(self):
        self.video.fps = self.jansen.screen.fps
        if self.detect:
            self.video.set_trajectories(search_range=self.link_tol,
                                        memory=int(self.nskip+3))
        if self.save_frames or self.save_trajectories:
            omit, omit_feat = ([], [])
            if not self.save_frames:
                omit.append('frames')
            if not self.save_trajectories:
                omit.append('trajectories')
            if not self.save_feature_data:
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

    @pyqtSlot(bool)
    def handleRealTime(self, selected):
        self.real_time = selected

    @pyqtSlot(bool)
    def handlePost(self, selected):
        self.real_time = not selected

    @pyqtSlot(bool)
    def handleSaveFrames(self, selected):
        self.save_frames = selected

    @pyqtSlot(bool)
    def handleSaveTrajectories(self, selected):
        self.save_trajectories = selected

    @pyqtSlot(bool)
    def handleSaveFeatureData(self, selected):
        self.save_feature_data = selected

    @pyqtSlot(int)
    def handleSkip(self, nskip):
        self.nskip = nskip

    @pyqtSlot(float)
    def handleLink(self, tol):
        self.link_tol = tol

    def post_process(self):
        self.jansen.screen.source.blockSignals(True)
        self.jansen.screen.pauseSignals(True)
        frames, detections = self.predict(self.frames,
                                          self.framenumbers,
                                          post=True)
        self.video.add(frames)
        self.frames = []
        self.framenumbers = []
        self.jansen.screen.source.blockSignals(False)
        self.jansen.screen.pauseSignals(False)

    def predict(self, images, framenumbers, post=False):
        frames = []
        detections = []
        for image in images:
            frames.append(Frame())
            detections.append([])
        return frames, detections

    def draw(self, detections):
        return []

    def remove(self, rois):
        if rois is not None:
            for rect in rois:
                self.jansen.screen.removeOverlay(rect)

    def init_pipeline(self):
        pass

    def clear_pipeline(self):
        pass
