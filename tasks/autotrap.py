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
        coords = list(map(lambda (x, y, w, h): QVector3D(x + w/2, y + h/2, self.parent.cgh.zc), rectangles))
        self.traps = self.parent.pattern.createTraps(coords)
