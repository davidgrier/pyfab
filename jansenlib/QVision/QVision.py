# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSlot
from .QVisionWidget import Ui_QVisionWidget
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QVision(QWidget):

    def __init__(self, parent=None, nskip=3):
        super(QVision, self).__init__(parent)
        self.ui = Ui_QVisionWidget()
        self.ui.setupUi(self)
        self.configurePlot()
        self._configuration = None
        self._detect = False
        self._estimate = False
        self._refine = False
        self._nskip = 3
        self._realTime = True
        self._saveFrames = False
        self._saveTrajectories = False
        self._discardEmpty = False
        self.configureUi()
        self.connectSignals()

    @property
    def configuration(self):
        return self._configuration

    @configuration.setter
    def configuration(self, config):
        self._configuration = config

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
        if self._detect:
            # TODO
            if self._estimate:
                # TODO
                if self._refine:
                    # TODO
                    pass

    @pyqtSlot(bool)
    def handleDetect(self, selected):
        self._detect = selected
        if selected:
            pass
            # detector =
        else:
            detector = None

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
