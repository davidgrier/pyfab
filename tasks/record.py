# -*- coding: utf-8 -*-

'''Task to record a video for a certain number of frames.'''

from task import task


class record(task):

    def __init__(self, fn=None, **kwargs):
        #pass in nframes keyword for length of recording 
        super(record, self).__init__(**kwargs)
        self.fn = fn
        
    def initialize(self, frame):
        if self.fn is not None:
            self.dvr = self.parent.dvr
            self.dvr.filename = self.fn
            self.dvr.brecord.animateClick()

    def dotask(self):
        if self.fn is not None:
            self.dvr.bstop.animateClick()
