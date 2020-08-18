# -*- coding: utf-8 -*-
# MENU: Max task

from .Task import Task
import numpy as np
import cv2


class MaxTask(Task):
    """Perform a task on a frame composed of the brightest pixels
    in nframes frames.

    By default, maxtask() saves the maximum-intensity image.
    Subclasses of maxtask() should override dotask() to perform
    operations based on the maximum-intensity image."""

    def __init__(self, prefix=None, nframes=20, **kwargs):
        super(MaxTask, self).__init__(nframes=nframes-1, **kwargs)
        self.prefix = prefix

    def initialize(self, frame):
        self.frame = frame

    def doprocess(self, frame):
        self.frame = np.maximum(frame, self.frame)

    def dotask(self):
        prefix = 'maxtask' if self.prefix is None else self.prefix
        fn = self.parent.configuration.filename(prefix=prefix,
                                                suffix='.png')
        cv2.imwrite(fn, self.frame)
