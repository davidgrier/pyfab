# -*- coding: utf-8 -*-

from .task import task
import numpy as np


class sethologram(task):
    def __init__(self, **kwargs):
        super(sethologram, self).__init__(delay=20, **kwargs)
        # Do not decrease delay value!
        self.kwargs = kwargs

    def initialize(self, frame):
        self.parent.pattern.clearTraps()
        self.cgh = self.parent.cgh
        self.qx = np.imag(self.cgh.iqx)
        self.qy = np.imag(self.cgh.iqy)

    def dotask(self):
        '''To set holograms to the SLM:
        1. Subclass sethologram
        2. Override dotask
        3. Run self.cgh.setPhi(phi) inside dotask, where phi
           is your custom hologram. 
        See vortex.py for a demonstration.
        '''
        pass
