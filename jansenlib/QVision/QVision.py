# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from .QVisionWidget import Ui_QVisionWidget

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

    @property
    def configuration(self):
        return self._configuration

    @configuration.setter
    def configuration(self, config):
        self._configuration = config

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, detect):
        self._detect = detect

    @property
    def estimate(self):
        return self._estimate

    @estimate.setter
    def estimate(self, estimate):
        self._estimate = estimate

    @property
    def refine(self):
        return self._refine

    @refine.setter
    def refine(self, refine):
        self._refine = refine

    def configurePlot(self):
        self.ui.plot.setBackground('w')
        self.ui.plot.getAxis('bottom').setPen(0.1)
        self.ui.plot.getAxis('left').setPen(0.1)
        self.ui.plot.showGrid(x=True, y=True)
        self.ui.plot.setLabel('bottom', 'a_p [um]')
        self.ui.plot.setLabel('left', 'n_p')
