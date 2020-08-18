# -*- coding: utf-8 -*-
# MENU: DVR/Start Recording

'''Task to start recording.'''

from ..QTask import QTask


class DVRStart(QTask):

    def __init__(self, fn=None, **kwargs):
        super(DVRStart, self).__init__(**kwargs)
        self.fn = fn

    def complete(self):
        self.dvr = self.parent().dvr
        if self.fn is not None:
            self.dvr.filename = self.fn
        self.dvr.recordButton.animateClick()
