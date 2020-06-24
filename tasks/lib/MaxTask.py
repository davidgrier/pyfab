# -*- coding: utf-8 -*-
# MENU: Acquire/Maximum Brightness

from ..QTask import QTask
import numpy as np
import cv2


class MaxTask(QTask):
    """
    Perform a task on a frame composed of the brightest pixels
    in nframes frames.

    By default, MaxTask() saves the maximum-intensity image to a png file.
    Subclasses of MaxTask() should override complete() to perform
    operations based on the maximum-intensity image.
    """

    def __init__(self, prefix='maxtask', nframes=20, **kwargs):
        super(MaxTask, self).__init__(nframes=nframes-1, **kwargs)
        self.prefix = prefix

    def initialize(self, frame):
        self.frame = frame

    def process(self, frame):
        self.frame = np.maximum(frame, self.frame)

    def complete(self):
        filename = self.parent().configuration.filename
        fn = filename(prefix=self.prefix, suffix='.png')
        cv2.imwrite(fn, self.frame)
