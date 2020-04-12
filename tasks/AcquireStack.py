# -*- coding: utf-8 -*-

'''
Base Task to acquire stack of either raw or pylorenzmie frames.
.'''

from .Task import Task


class AcquireStack(Task):

    '''Acquires nframes of either raw or pylorenzmie frames.'''

    def __init__(self, **kwargs):
        super(AcquireStack, self).__init__(**kwargs)
        self.frames = []

    def doprocess(self, frame):
        self.frames.append(frame)
