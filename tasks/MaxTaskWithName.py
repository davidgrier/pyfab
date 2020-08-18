# -*- coding: utf-8 -*-

from .MaxTask import MaxTask
import numpy as np
import cv2


class MaxTaskWithName(MaxTask):

    def __init__(self, fn=None, nframes=20, **kwargs):
        super(MaxTaskWithName, self).__init__(nframes=nframes - 1, **kwargs)
        self.fn = fn

    def initialize(self, frame):
        self.frame = frame

    def doprocess(self, frame):
        self.frame = np.maximum(frame, self.frame)

    def dotask(self):
        if self.fn is not None:
            cv2.imwrite(self.parent.configuration.filename(prefix=self.fn,suffix='.png'), self.frame)
        else:
            cv2.imwrite(self.parent.configuration.filename(prefix='MaxTask',suffix='.png'), self.frame)
