# -*- coding: utf-8 -*-

'''Provides functionality for particle localization,
including particle detection and tracking video filter'''

from pyqtgraph.Qt import QtCore
from .detect import bytscl, Detector
import pyqtgraph as pg


class QDetector(QtCore.QObject):

    def __init__(self, parent=None):
        super(QDetector, self).__init__()
        self.detector = Detector(cascade_fn='cascade_example.xml')
        self.parent = parent
        self.rois = []
        self.capacity = 5
        self.min_neighbors = 45
        self.scale_factor = 1.1

    def _init_rects(self, expand_factor=1):
        for idx in range(self.capacity * expand_factor):
            rect = pg.RectROI([1000, 1000], [0, 0], pen=(3, 1))
            self.rois.append(rect)
            self.parent.addOverlay(rect)

    def _draw(self, rectangles):
        expand_factor = len(rectangles) // len(self.rois)
        if expand_factor >= 1:
            self.init_rects(expand_factor=expand_factor)
        for idx, (x, y, w, h) in enumerate(rectangles):
            self.rois[idx].setPos([x, y])
            self.rois[idx].setSize([w, h])

    def _retreat(self):
        for roi in self.rois:
            if roi.size() != (0, 0):
                roi.setPos([1000, 1000])
                roi.setSize([0, 0])

    def remove(self):
        '''Destroys all rectangles from screen
        '''
        for rect in self.rois:
            self.parent.removeOverlay(rect)
        self.rois = []

    def detect(self, frame):
        '''Filter for drawing rectangles on particle location
        '''
        if len(self.rois) == 0:
            self._init_rects()
        self._retreat()
        rectangles = self.detector.detect(bytscl(frame*1.2),
                                          min_neighbors=self.min_neighbors,
                                          scale_factor=self.scale_factor)
        self._draw(rectangles)
        return frame

    def grab(self, frame):
        '''A method for returning particle location in a single frame
        '''
        return self.detector.detect(bytscl(frame*1.2),
                                    min_neighbors=self.min_neighbors,
                                    scale_factor=self.scale_factor)
