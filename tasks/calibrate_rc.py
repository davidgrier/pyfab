# -*- coding: utf-8 -*-

from maxtask import maxtask
import trackpy as tp


class calibrate_rc(maxtask):
    """Locate the zero-th order spot in the camera image.

    The zero-th order should be brought to a focus before
    performing this task."""

    def __init__(self, **kwargs):
        super(calibrate_rc, self).__init__(**kwargs)

    def initialize(self, frame):
        self.parent.pattern.clearTraps()

    def dotask(self):
        f = tp.locate(self.frame, 11, topn=1, characterize=False)
        self.parent.wcgh.xc = f['x']
        self.parent.wcgh.yc = f['y']
