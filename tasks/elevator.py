# -*- coding: utf-8 -*-
# MENU: Experiments/Measure z continuous

from .movetocoordinate import movetocoordinate
import pandas as pd


class elevator(movetocoordinate):
    """Automatically move traps up then down in the z direction."""

    def __init__(self, **kwargs):
        super(elevator, self).__init__(z=-200,
                                       travel_back=True,
                                       correct=False,
                                       **kwargs)

    def initialize_more(self, frame):
        self.dvr = self.parent.dvr
        self.dvr.recordButton.animateClick()
        self.trap_list = self.parent.pattern.pattern.flatten()
        self.frames = []
        self.data = {}
        for trap in self.trap_list:
            self.data[trap] = []

    def process_more(self, frame):
        self.frames.append(self.dvr.framenumber)
        for trap in self.trap_list:
            self.data[trap].append(trap.r.z())

    def dotask(self):
        self.dvr.stopButton.animateClick()
        df = pd.DataFrame(index=self.frames, data=self.data)
        df.to_csv(self.dvr.filename[:-4] + ".csv")
