# -*- coding: utf-8 -*-

'''Task to record a video for a certain number of frames.'''

from .task import task


class record(task):

    def __init__(self, fn=None, **kwargs):
        # Pass in nframes keyword for length of recording 
        super(record, self).__init__(**kwargs)
        self.fn = fn
        
    def initialize(self, frame):
        self.dvr = self.parent.dvr
        if self.fn is not None:
            self.dvr.filename = self.fn
        self.dvr.recordButton.animateClick()

    def dotask(self):
        self.dvr.stopButton.animateClick()
