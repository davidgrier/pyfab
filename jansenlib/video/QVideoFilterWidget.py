# -*- coding: utf-8 -*-

"""Control panel for standard video filters."""

from pyqtgraph.Qt import QtGui, QtCore
from .vmedian import vmedian
from .vmax import vmax
from .QDetector import QDetector
import numpy as np
import cv2
from matplotlib.pylab import cm
try:
    from PyQt4.QtCore import QString
except ImportError:
    QString = str


class QVideoFilterWidget(QtGui.QFrame):

    def __init__(self, parent):
        super(QVideoFilterWidget, self).__init__(parent)
        video = self.parent().screen.video
        self.register = video.registerFilter
        self.unregister = video.unregisterFilter
        self.init_filters()
        self.init_ui()

    def init_filters(self):
        self.median = vmedian(order=3)
        self.deflicker = vmax(order=4)
        self.detector = QDetector(parent=self.parent().screen)

    def init_ui(self):
        self.setFrameShape(QtGui.QFrame.Box)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(1)
        title = QtGui.QLabel('Video Filters')
        bmedian = QtGui.QCheckBox(QString('Median'))
        bdeflicker = QtGui.QCheckBox(QString('Deflicker'))
        bnormalize = QtGui.QCheckBox(QString('Normalize'))
        bsample = QtGui.QCheckBox(QString('Sample and Hold'))
        bndvi = QtGui.QCheckBox(QString('NDVI'))
        bdetect = QtGui.QCheckBox(QString('Detect'))
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

    @QtCore.pyqtSlot(bool)
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

    @QtCore.pyqtSlot(bool)
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
        nrm = (frame.astype(float) - 13) / (self.background - 13)
        n = np.clip(100 * nrm, 0, 255).astype(np.uint8)
        if np.amax(n) == 255:
            print('There is saturation!' + np.random.choice(['!', '?']))
        return n

    @QtCore.pyqtSlot(bool)
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

    @QtCore.pyqtSlot(bool)
    def handleNDVI(self, selected):
        if selected:
            self.register(self.ndvi)
        else:
            self.unregister(self.ndvi)

    @QtCore.pyqtSlot(bool)
    def handleDetect(self, selected):
        if selected:
            self.register(self.detector.detect)
        else:
            self.unregister(self.detector.detect)
            self.detector.remove()
