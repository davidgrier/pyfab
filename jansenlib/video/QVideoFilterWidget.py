# -*- coding: utf-8 -*-

"""Control panel for standard video filters."""

from pyqtgraph.Qt import QtGui, QtCore
from vmedian import vmedian
import numpy as np
import cv2
from matplotlib.pylab import cm
from detect import bytscl, Detector

detector = Detector(cascade_fn='cascade_example.xml')

class QVideoFilterWidget(QtGui.QFrame):

    def __init__(self, video):
        super(QVideoFilterWidget, self).__init__()
        self.video = video
        self.height = self.video.source.size.height()
        self.scene = self.video.parent.scene()
        self.init_filters()
        self.init_ui()

    def init_filters(self):
        self.median = vmedian(order=3)

    def init_ui(self):
        self.setFrameShape(QtGui.QFrame.Box)
        layout = QtGui.QVBoxLayout(self)
        layout.setMargin(1)
        title = QtGui.QLabel('Video Filters')
        bmedian = QtGui.QCheckBox(QtCore.QString('Median'))
        bnormalize = QtGui.QCheckBox(QtCore.QString('Normalize'))
        bsample = QtGui.QCheckBox(QtCore.QString('Sample and Hold'))
        bndvi = QtGui.QCheckBox(QtCore.QString('NDVI'))
        bdetect = QtGui.QCheckBox(QtCore.QString('Detect'))
        layout.addWidget(title)
        layout.addWidget(bmedian)
        layout.addWidget(bnormalize)
        layout.addWidget(bsample)
        layout.addWidget(bndvi)
        layout.addWidget(bdetect)
        bmedian.clicked.connect(self.handleMedian)
        bnormalize.clicked.connect(self.handleNormalize)
        bsample.clicked.connect(self.handleSample)
        bndvi.clicked.connect(self.handleNDVI)
        bdetect.clicked.connect(self.handleDetect)

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
            
              
    def detect(self, frame):
	self.scene.removeItem(self.rectgroup)
        self.rectgroup = QtGui.QGraphicsItemGroup(scene=self.scene)
        rectangles = detector.detect(bytscl(frame*1.2), min_neighbors=45, scale_factor=1.1)
        map(lambda (x,y,w,h): self.rectgroup.addToGroup(QtGui.QGraphicsRectItem(x, self.height - (y+h/2.), w, h)), rectangles)        
        return frame
		
	
    @QtCore.pyqtSlot(bool)
    def handleDetect(self, selected):
        if selected:
            self.rectgroup = QtGui.QGraphicsItemGroup(scene=self.scene)
            self.video.registerFilter(self.detect)
        else:
            self.scene.removeItem(self.rectgroup)
            self.video.unregisterFilter(self.detect)
		
