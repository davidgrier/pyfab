# -*- coding: utf-8 -*-
# MENU: Experiments/Modify Phase Ultra Trap (Image)

from .task import task
import numpy as np
from pyfablib.traps.QUltraTrap import QUltraTrap
import os
from pyqtgraph.Qt import QtGui


class ModifyPhaseUltraTrapImage(task):
    """Delay, record, and move traps in the z direction by changing their hologram."""

    def __init__(self, measure_bg=False, **kwargs):
        super(ModifyPhaseUltraTrapImage, self).__init__(**kwargs)
        self.traps = None

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        for t in self.traps.flatten(): 
              if isinstance(t, QUltraTrap): self.trap = t; # selecting the ultra traps
        self.deltaZ = 30; self.DdeltaPhi = np.pi/10; 
        # The deltaPhi of ultra trap changes by DdeltaPhi in each step while its separation is deltaZ.
                
    def dotask(self):
        self.traps = self.parent.pattern.pattern
        if isinstance(self.trap, QUltraTrap):
            fn0, fn_ext = os.path.splitext(self.parent.dvr.filename)
            self.register('Modify', group=self.traps, NewDeltaZ=self.deltaZ, NewDeltaPhi=0)
            for n in range(0, int(np.absolute((2*np.pi)/self.DdeltaPhi))+1 ): 
                dPhinew = np.absolute(self.DdeltaPhi*n) 
                self.register('Delay', delay=50)
                self.register('MaxTaskWithName',fn='{:02d}'.format(n))
                self.register('Delay', delay=10)
                self.register('Modify', group=self.traps, NewDeltaZ=self.deltaZ, NewDeltaPhi=dPhinew)
