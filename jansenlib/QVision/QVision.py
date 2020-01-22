# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot
from .QVisionWidget import Ui_QVisionWidget

from CNNLorenzMie.Localizer import Localizer
from pylorenzmie.theory.Video import Video
from pylorenzmie.theory.Frame import Frame
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import coordinates

import numpy as np
import pyqtgraph as pg

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QVision(QWidget):

    def __init__(self, parent=None, nskip=3):
        super(QVision, self).__init__(parent)
        self.ui = Ui_QVisionWidget()
        self.ui.setupUi(self)

        self.jansen = None

        self.model = LMHologram()
        self.model.double_precision = False
        self.model.coordinates = None

        self.video = Video()

        self.detect = False
        self.estimate = False
        self.refine = False
        self.nskip = 3
        self.counter = self.nskip
        self.real_time = True
        self.save_frames = False
        self.save_trajectories = False
        self.discard_empty = False

        self.rois = None

        self.configurePlot()
        self.configureUi()
        self.connectSignals()

    def connectSignals(self):
        self.ui.breal.toggled.connect(self.handleRealTime)
        self.ui.bpost.toggled.connect(self.handlePost)
        self.ui.checkDiscard.toggled.connect(self.handleDiscard)
        self.ui.checkFrames.clicked.connect(self.handleSaveFrames)
        self.ui.checkTrajectories.clicked.connect(self.handleSaveTrajectories)
        self.ui.bDetect.clicked.connect(self.handleDetect)
        self.ui.bEstimate.clicked.connect(self.handleEstimate)
        self.ui.bRefine.clicked.connect(self.handleRefine)
        self.ui.skipBox.valueChanged.connect(self.handleSkip)

    def configureUi(self):
        self.ui.bDetect.setChecked(self.detect)
        self.ui.bEstimate.setChecked(self.estimate)
        self.ui.bRefine.setChecked(self.refine)
        self.ui.checkFrames.setChecked(self.save_frames)
        self.ui.checkTrajectories.setChecked(self.save_trajectories)
        self.ui.breal.setChecked(self.real_time)
        self.ui.checkDiscard.setChecked(self.discard_empty)
        self.ui.skipBox.setProperty("value", self.nskip)

    def configurePlot(self):
        self.ui.plot.setBackground('w')
        self.ui.plot.getAxis('bottom').setPen(0.1)
        self.ui.plot.getAxis('left').setPen(0.1)
        self.ui.plot.showGrid(x=True, y=True)
        self.ui.plot.setLabel('bottom', 'a_p [um]')
        self.ui.plot.setLabel('left', 'n_p')

    @pyqtSlot(np.ndarray)
    def process(self, image):
        self.remove(self.rois)
        if self.counter == 0:
            self.counter = self.nskip
            detections = []
            estimations = None
            if self.detect:
                if len(image.shape) == 2:
                    inflated = np.stack((image,)*3, axis=-1)
                else:
                    inflated = image
                detections = self.localizer.predict(img_list=[inflated])[0]
                self.rois = self.draw(detections)
                if self.estimate:
                    estimations = None
                else:
                    estimations = None
            frame = self.build(image, detections, estimations)
            if self.jansen.dvr.is_recording() and self.save_frames:
                if not (self.discard_empty and len(detections) == 0):
                    self.video.add(frame)
                    print("Frame added")
        else:
            self.counter -= 1
            self.rois = None

    def build(self, image, detections, estimations):
        features = []
        for idx, detection in enumerate(detections):
            x, y, l, w = detection['bbox']
            xc, yc = (x-w//2, y+l//2)
            self.model.coordinates = coordinates((w, l),
                                                 corner=(xc, yc))
            self.model.x_p = x
            self.model.y_p = y
            feature = Feature(self.model)
            feature.amoeba_settings.options['maxevals'] = 500
            if estimations is not None:
                pass
            features.append(feature)
        if self.jansen.dvr.recording:
            i = self.jansen.dvr.framenumber
        else:
            i = None
        frame = Frame(data=image, features=features, framenumber=i)
        if self.refine:
            frame.optimize(method='amoeba')
        return frame

    def draw(self, detections):
        rois = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            roi = pg.RectROI([x, y], [w, h], pen=(3, 1))
            self.jansen.screen.addOverlay(roi)
            rois.append(roi)
        return rois

    def remove(self, rois):
        if rois is not None:
            for rect in rois:
                self.jansen.screen.removeOverlay(rect)

    def save(self):
        if self.save_frames:
            if self.save_trajectories:
                self.video.set_trajectories()
                omit = []
            else:
                omit = ['trajectories']
            filename = self.jansen.dvr.filename.split(".")[0] + '.json'
            self.video.serialize(filename=filename,
                                 omit=omit, omit_feat=['data'])
            logger.info("{} saved.".format(filename))
            self.video = Video()

    @pyqtSlot(bool)
    def handleDetect(self, selected):
        if selected:
            self.detect = True
            self.estimate = False
            self.refine = False
            self.localizer = Localizer(configuration='tinyholo',
                                       weights='_500k')
        else:
            self.detect = False
            self.estimate = False
            self.refine = False
            self.localizer = None

    @pyqtSlot(bool)
    def handleEstimate(self, selected):
        if selected:
            self.detect = True
            self.estimate = True
            self.refine = False
        else:
            self.detect = False
            self.estimate = False
            self.refine = False

    @pyqtSlot(bool)
    def handleRefine(self, selected):
        if selected:
            self.detect = True
            self.estimate = True
            self.refine = True
        else:
            self.detect = False
            self.estimate = False
            self.refine = False

    @pyqtSlot(bool)
    def handleRealTime(self, selected):
        self.real_time = selected

    @pyqtSlot(bool)
    def handlePost(self, selected):
        self.real_time = not selected

    @pyqtSlot(bool)
    def handleDiscard(self, selected):
        self.discard_empty = selected

    @pyqtSlot(bool)
    def handleSaveFrames(self, selected):
        self.save_frames = selected

    @pyqtSlot(bool)
    def handleSaveTrajectories(self, selected):
        self.save_trajectories = selected

    @pyqtSlot(int)
    def handleSkip(self, nskip):
        self.nskip = nskip
