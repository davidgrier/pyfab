# -*- coding: utf-8 -*-

from .task import task
import numpy as np
import cv2


class maxtask(task):
    """Perform a task on a frame composed of the brightest pixels
    in nframes frames.

    By default, maxtask() saves the maximum-intensity image.
    Subclasses of maxtask() should override dotask() to perform
    operations based on the maximum-intensity image."""

    def __init__(self, nframes=10, **kwargs):
        super(maxtask, self).__init__(nframes=nframes - 1, **kwargs)

    def initialize(self, frame):
        self.frame = frame

    def doprocess(self, frame):
        self.frame = np.maximum(frame, self.frame)

    def dotask(self):
        fn = self.parent.config.filename(prefix='maxtask', suffix='.png')
        cv2.imwrite(fn, self.frame)
