from PyQt4 import QtGui, QtCore
from vmedian import vmedian
import numpy as np
import cv2
from matplotlib.pylab import cm


class QFabFilter(QtGui.QFrame):

    def __init__(self, video):
        super(QFabFilter, self).__init__()
        self.video = video
        self.init_filters()
        self.init_ui()

    def init_filters(self):
        shape = self.video.image.shape
        self.median = vmedian(order=3, shape=shape)

    def init_ui(self):
        self.setFrameShape(QtGui.QFrame.Box)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(1)
        title = QtGui.QLabel('Video Filters')
        bmedian = QtGui.QCheckBox(QtCore.QString('Median'))
        bnormalize = QtGui.QCheckBox(QtCore.QString('Normalize'))
        bsample = QtGui.QCheckBox(QtCore.QString('Sample and Hold'))
        bndvi = QtGui.QCheckBox(QtCore.QString('NDVI'))
        layout.addWidget(title)
        layout.addWidget(bmedian)
        layout.addWidget(bnormalize)
        layout.addWidget(bsample)
        layout.addWidget(bndvi)
        bmedian.clicked.connect(self.handleMedian)
        bnormalize.clicked.connect(self.handleNormalize)
        bsample.clicked.connect(self.handleSample)
        bndvi.clicked.connect(self.handleNDVI)

    @QtCore.pyqtSlot(bool)
    def handleMedian(self, selected):
        if selected:
            self.video.registerFilter(self.median.filter)
        else:
            self.video.unregisterFilter(self.median.filter)

    def normalize(self, frame):
        self.median.add(frame)
        med = self.median.get()
        med = np.clip(med, 1, 255)
        nrm = frame.astype(float) / med
        return np.clip(100 * nrm, 0, 255).astype(np.uint8)

    @QtCore.pyqtSlot(bool)
    def handleNormalize(self, selected):
        if selected:
            self.video.registerFilter(self.normalize)
        else:
            self.video.unregisterFilter(self.normalize)

    def samplehold(self, frame):
        if not frame.shape == self.median.shape:
            self.median.add(frame)
        if not self.median.initialized:
            self.median.add(frame)
            self.background = np.clip(self.median.get(), 1, 255)
        nrm = frame.astype(float) / self.background
        return np.clip(100 * nrm, 0, 255).astype(np.uint8)

    @QtCore.pyqtSlot(bool)
    def handleSample(self, selected):
        if selected:
            self.median.reset()
            self.video.registerFilter(self.samplehold)
        else:
            self.video.unregisterFilter(self.samplehold)

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
            self.video.registerFilter(self.ndvi)
        else:
            self.video.unregisterFilter(self.ndvi)
