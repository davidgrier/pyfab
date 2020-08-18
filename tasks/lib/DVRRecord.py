# -*- coding: utf-8 -*-
# MENU: DVR/Record

'''Task to record a specified number of frames.'''

from ..QTask import QTask


class DVRRecord(QTask):

    def __init__(self, fn=None, nframes=30, **kwargs):
        super(DVRRecord, self).__init__(nframes=nframes, **kwargs)
        self.fn = fn

    def initialize(self, frame):
        self.dvr = self.parent().dvr
        if self.fn is not None:
            self.dvr.filename = self.fn
        self.dvr.recordButton.animateClick()

    def complete(self):
        self.dvr.stopButton.animateClick()
