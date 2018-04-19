# -*- coding: utf-8 -*-

'''Stores detector for automated trapping and provides routine for adding rectangular overlays to detected particles.'''

from pyqtgraph.Qt import QtCore, QtGui
from detect import bytscl, Detector
import pyqtgraph as pg
import numpy as np

class QDetector(QtCore.QObject):

    giveRects = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super(QDetector, self).__init__()
        self.detector = Detector(cascade_fn='cascade_example.xml')
        self.parent = parent
        self.rois = []

    def addRects(self, rectangles):
        for (x,y,w,h) in rectangles:
            rect = pg.RectROI([x+w, y+h], [w, h], pen=(3,1))
            self.rois.append(rect)
            self.parent.addOverlay(rect)
    
    def removeRects(self):
        for rect in self.rois:
            self.parent.removeOverlay(rect)
        self.rois = []

    def detect(self, frame):
        self.removeRects()
        rectangles = self.detector.detect(bytscl(frame*1.2), min_neighbors=45, scale_factor=1.1)
        self.addRects(rectangles)
        return frame

    def grab(self, frame):
        return self.detector.detect(bytscl(frame*1.2), min_neighbors=45, scale_factor=1.1)
