# -*- coding: utf-8 -*-

from .Task import Task
import numpy as np

## is setHologram outdated?
class SetHologram(Task):
    def __init__(self, **kwargs):
        super(SetHologram, self).__init__(delay=20, **kwargs)
        # Do not decrease delay value!
        self.kwargs = kwargs

    def initialize(self, frame):
        self.parent.pattern.clearTraps()
        self.cgh = self.parent.cgh.device
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
