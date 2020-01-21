# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot
from .QVisionWidget import Ui_QVisionWidget

from CNNLorenzMie.Localizer import Localizer
from pylorenzmie.theory.Video import Video
from pylorenzmie.theory.Frame import Frame
from pylorenzmie.theory.Feature import Feature
from pylorenzmie.theory.LMHologram import LMHologram
from pylorenzmie.theory.Instrument import Instrument, coordinates

import numpy as np
import pyqtgraph as pg

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QVision(QWidget):

    def __init__(self, parent=None, nskip=3):
        super(QVision, self).__init__(parent)
        self.ui = Ui_QVisionWidget()
        self.ui.setupUi(self)

        self.parent = parent
        self.instrument = Instrument()
        self.model = LMHologram()
        self.model.double_precision = False
        self.model.coordinates = None

        self.frameinfo = []
        self.video = None

        self._detect = False
        self._estimate = False
        self._refine = False
        self._nskip = 3
        self._counter = self._nskip
        self._realTime = True
        self._saveFrames = False
        self._saveTrajectories = False
        self._discardEmpty = False

        self.rois = None

        self.configurePlot()
        self.configureUi()
        self.connectSignals()

    def connectSignals(self):
        self.ui.breal.toggled.connect(self.handleRealTime)
        self.ui.bpost.toggled.connect(self.handlePost)
        self.ui.bdiscard.toggled.connect(self.handleDiscard)
        self.ui.checkFrames.clicked.connect(self.handleSaveFrames)
        self.ui.checkTrajectories.clicked.connect(self.handleSaveTrajectories)
        self.ui.checkDetect.clicked.connect(self.handleDetect)
        self.ui.checkEstimate.clicked.connect(self.handleEstimate)
        self.ui.checkRefine.clicked.connect(self.handleRefine)
        self.ui.skipBox.valueChanged.connect(self.handleSkip)

    def configureUi(self):
        self.ui.checkDetect.setChecked(self._detect)
        self.ui.checkEstimate.setChecked(self._estimate)
        self.ui.checkRefine.setChecked(self._refine)
        self.ui.checkFrames.setChecked(self._saveFrames)
        self.ui.checkTrajectories.setChecked(self._saveTrajectories)
        self.ui.breal.setChecked(self._realTime)
        self.ui.bdiscard.setChecked(self._discardEmpty)
        self.ui.skipBox.setProperty("value", self._nskip)

    def configurePlot(self):
        self.ui.plot.setBackground('w')
        self.ui.plot.getAxis('bottom').setPen(0.1)
        self.ui.plot.getAxis('left').setPen(0.1)
        self.ui.plot.showGrid(x=True, y=True)
        self.ui.plot.setLabel('bottom', 'a_p [um]')
        self.ui.plot.setLabel('left', 'n_p')

    @pyqtSlot(np.ndarray)
    def process(self, frame):
        self.remove(self.rois)
        if self._counter == 0:
            self._counter = self._nskip
            if self._detect:
                detections = self.localizer.predict(img_list=[frame])
                self.rois = self.draw(detections[0])
                if self._estimate:
                    estimations = None
                else:
                    estimations = None
                lmframe = self.build(
                    frame, detections[0], estimations)
                if self._saveFrames:
                    if not (self._discardEmpty and len(detections) == 0):
                        self.video.appendFrame(lmframe)
        else:
            self._counter -= 1
            self.rois = None

    def build(self, frame, detections, estimations):
        features = []
        for idx, detection in enumerate(detections):
            x, y, w, h = detection['bbox']
            xc, yc = (x-w//2, y+h//2)
            self.model.coordinates = coordinates((w, h),
                                                 corner=(xc, yc))
            self.model.x_p = x
            self.model.y_p = y
            feature = Feature(self.model)
            feature.amoeba_settings.options['maxevals'] = 500
            if estimations is not None:
                pass
            features.append(feature)
        f = Frame(data=frame, features=features)
        if self._refine:
            f.optimize(method='amoeba')
        return f

    def draw(self, detections):
        rois = []
        for detection in detections:
            x, y, w, h = detection['bbox']
            roi = pg.RectROI([x, y], [w, h], pen=(3, 1))
            self.parent.screen.addOverlay(roi)
            rois.append(roi)
        return rois

    def remove(self, rois):
        if rois is not None:
            for rect in rois:
                self.parent.screen.removeOverlay(rect)

    @pyqtSlot(bool)
    def handleDetect(self, selected):
        self._detect = selected
        if selected:
            self.localizer = Localizer(configuration='tinyholo',
                                       weights='_500k')
        else:
            self.localizer = None

    @pyqtSlot(bool)
    def handleEstimate(self, selected):
        self._estimate = selected

    @pyqtSlot(bool)
    def handleRefine(self, selected):
        self._refine = selected

    @pyqtSlot(bool)
    def handleRealTime(self, selected):
        self._realTime = selected

    @pyqtSlot(bool)
    def handlePost(self, selected):
        self._realTime = not selected

    @pyqtSlot(bool)
    def handleDiscard(self, selected):
        self._discardEmpty = selected

    @pyqtSlot(bool)
    def handleSaveFrames(self, selected):
        self._saveFrames = selected

    @pyqtSlot(bool)
    def handleSaveTrajectories(self, selected):
        self._saveTrajectories = selected

    @pyqtSlot(int)
    def handleSkip(self, nskip):
        self._nskip = nskip
