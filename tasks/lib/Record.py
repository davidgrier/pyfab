# -*- coding: utf-8 -*-

'''Task to record a video for a certain number of frames.'''

from ..QTask import QTask


class Record(QTask):

    def __init__(self, fn=None, nframes=30, **kwargs):
        # Pass in nframes keyword for length of recording
        super(Record, self).__init__(nframes=nframes, **kwargs)
        self.fn = fn

    def initialize(self, frame):
        self.dvr = self.parent().dvr
        if self.fn is not None:
            self.dvr.filename = self.fn
        self.dvr.recordButton.animateClick()

    def complete(self):
        if self.stop:
            self.dvr.stopButton.animateClick()
