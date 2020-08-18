# -*- coding: utf-8 -*-

from .Task import Task


class Wait(Task):

    '''
    Wait for particle assembly to finish.
    '''

    def __init__(self, **kwargs):
        super(Wait, self).__init__(**kwargs)
        self.nframes = 1e6

    def initialize(self, frame):
        self.assembler = self.parent.assembler

    def doprocess(self, frame):
        if not self.assembler.running:
            self.nframes = 0
