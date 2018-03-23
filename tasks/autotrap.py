import numpy as np
from jansenlib.video.detect import bytscl, Detector
from task import task
from PyQt4.QtGui import QVector3D

detector = Detector(cascade_fn='cascade_example.xml')

class autotrap(task):

    def __init__(self, **kwargs):
        super(autotrap, self).__init__(nframes=1, **kwargs)

    def doprocess(self, frame):
        rectangles = detector.detect(bytscl(frame*1.2), min_neighbors=23)
        coords = list(map(lambda (x,y,w,h): QVector3D(x + w/2, y + h/2, self.parent.cgh.zc), rectangles))
        self.parent.pattern.createTraps(coords)
