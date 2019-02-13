# -*- coding: utf-8 -*-

"""Control panel for standard video filters."""

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QFrame, QVBoxLayout, QLabel, QCheckBox)
from .vmedian import vmedian
from .vmax import vmax
from .QDetector import QDetector
import numpy as np
import cv2
from matplotlib.pylab import cm


class QVideoFilterWidget(QFrame):

    def __init__(self, parent):
        super(QVideoFilterWidget, self).__init__(parent)
        video = self.parent().screen
        self.register = video.registerFilter
        self.unregister = video.unregisterFilter
        self.init_filters()
        self.init_ui()

    def init_filters(self):
        self.median = vmedian(order=3)
        self.deflicker = vmax(order=4)
        self.detector = QDetector(parent=self.parent().screen)

    def init_ui(self):
        self.setFrameShape(QFrame.Box)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(1, 1, 1, 1)
        title = QLabel('Video Filters')
        title.setFont(QFont('Arial', weight=QFont.Bold))
        bmedian = QCheckBox('Median')
        bdeflicker = QCheckBox('Deflicker')
        bnormalize = QCheckBox('Normalize')
        bsample = QCheckBox('Sample and Hold')
        bndvi = QCheckBox('NDVI')
        bdetect = QCheckBox('Detect')
        layout.addWidget(title)
        layout.addWidget(bmedian)
        layout.addWidget(bdeflicker)
        layout.addWidget(bnormalize)
        layout.addWidget(bsample)
        layout.addWidget(bndvi)
        layout.addWidget(bdetect)
        bmedian.clicked.connect(self.handleMedian)
        bdeflicker.clicked.connect(self.handleDeflicker)
        bnormalize.clicked.connect(self.handleNormalize)
        bsample.clicked.connect(self.handleSample)
        bndvi.clicked.connect(self.handleNDVI)
        bdetect.clicked.connect(self.handleDetect)

    @pyqtSlot(bool)
    def handleMedian(self, selected):
        if selected:
            self.register(self.median.filter)
        else:
            self.unregister(self.median.filter)

    def handleDeflicker(self, selected):
        if selected:
            self.register(self.deflicker.filter)
        else:
            self.unregister(self.deflicker.filter)

    def normalize(self, frame):
        self.median.add(frame)
        med = self.median.get()
        med = np.clip(med, 1, 255)
        nrm = frame.astype(float) / med
        return np.clip(100 * nrm, 0, 255).astype(np.uint8)

    @pyqtSlot(bool)
    def handleNormalize(self, selected):
        if selected:
            self.register(self.normalize)
        else:
            self.unregister(self.normalize)

    def samplehold(self, frame):
        if not frame.shape == self.median.shape:
            self.median.add(frame)
        if not self.median.initialized:
            self.median.add(frame)
            self.background = np.clip(self.median.get(), 1, 255)
        nrm = frame.astype(float) / self.background
        return np.clip(100 * nrm, 0, 255).astype(np.uint8)

    @pyqtSlot(bool)
    def handleSample(self, selected):
        if selected:
            self.median.reset()
            self.register(self.samplehold)
        else:
            self.unregister(self.samplehold)

    def ndvi(self, frame):
        if frame.ndim == 3:
            (r, g, b) = cv2.split(frame)
            r = r.astype(float)
            g = g.astype(float)
            ndx = (r - g) / np.clip(r + g, 1., 255)
            ndx = np.clip(128. * ndx + 127., 0, 255).astype(np.uint8)
        else:
            ndx = frame
        ndx = cm.RdYlGn_r(ndx, bytes=True)
        ndx = cv2.cvtColor(ndx, cv2.COLOR_BGRA2BGR)
        return ndx

    @pyqtSlot(bool)
    def handleNDVI(self, selected):
        if selected:
            self.register(self.ndvi)
        else:
            self.unregister(self.ndvi)

    @pyqtSlot(bool)
    def handleDetect(self, selected):
        if selected:
            self.register(self.detector.detect)
        else:
            self.unregister(self.detector.detect)
            self.detector.remove()
