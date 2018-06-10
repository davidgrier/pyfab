# -*- coding: utf-8 -*-
# MENU: Auto-Trap

from task import task
from PyQt4.QtGui import QVector3D


class autotrap(task):
    """Detect and trap particles on the screen."""

    def __init__(self, **kwargs):
        super(autotrap, self).__init__(**kwargs)
        self.traps = None

    def initialize(self, frame):
        rectangles = self.parent.detector.grab(frame)
        coords = list(map(lambda feature: QVector3D(feature[0] + feature[2]/2,
                            feature[1] + feature[3]/2, self.parent.cgh.zc), rectangles))
        self.traps = self.parent.pattern.createTraps(coords)
