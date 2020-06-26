# -*- coding: utf-8 -*-
# MENU: DVR/Stop Recording

'''Task to stop recording.'''

from ..QTask import QTask


class DVRStop(QTask):

    def __init__(self, **kwargs):
        super(DVRStop, self).__init__(**kwargs)

    def complete(self):
        self.parent().dvr.stopButton.animateClick()
