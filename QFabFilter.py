from PyQt4 import QtGui, QtCore
from vmedian import vmedian
import numpy as np


class QFabFilter(QtGui.QFrame):

    def __init__(self, video):
        super(QFabFilter, self).__init__()
        self.video = video
        self.init_filters()
        self.init_ui()

    def init_filters(self):
        shape = self.video.shape()
        self.median = vmedian(order=3, shape=shape)

    def init_ui(self):
        self.setFrameShape(QtGui.QFrame.Box)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(1)
        title = QtGui.QLabel('Video Filters')
        bmedian = QtGui.QCheckBox(QtCore.QString('Median'))
        bnormalize = QtGui.QCheckBox(QtCore.QString('Normalize'))
        bsample = QtGui.QCheckBox(QtCore.QString('Sample and Hold'))
        layout.addWidget(title)
        layout.addWidget(bmedian)
        layout.addWidget(bnormalize)
        layout.addWidget(bsample)
        bmedian.clicked.connect(self.handleMedian)
        bnormalize.clicked.connect(self.handleNormalize)
        bsample.clicked.connect(self.handleSample)

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

    def handleNormalize(self, selected):
        if selected:
            self.video.registerFilter(self.normalize)
        else:
            self.video.unregisterFilter(self.normalize)

    def samplehold(self, frame):
        if not self.median.initialized:
            self.median.add(frame)
            self.background = np.clip(self.median.get(), 1, 255)
        nrm = frame.astype(float) / self.background
        return np.clip(100 * nrm, 0, 255).astype(np.uint8)

    def handleSample(self, selected):
        if selected:
            self.median.reset()
            self.video.registerFilter(self.samplehold)
        else:
            self.video.unregisterFilter(self.samplehold)
