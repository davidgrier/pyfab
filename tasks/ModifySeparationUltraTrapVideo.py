# -*- coding: utf-8 -*-
# MENU: Experiments/Modify Separation Ultra Trap (Video)

from .task import task
import numpy as np
from pyfablib.traps.QUltraTrap import QUltraTrap
import os
from pyqtgraph.Qt import QtGui


class ModifySeparationUltraTrapVideo(task):
    """Delay, record, and move traps in the z direction by changing their hologram."""

    def __init__(self, measure_bg=False, **kwargs):
        super(ModifySeparationUltraTrapVideo, self).__init__(**kwargs)
        self.traps = None

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        for t in self.traps.flatten(): 
              if isinstance(t, QUltraTrap): self.trap = t; # selecting the ultra traps
        self.DdeltaZ = -1; self.FirstDeltaZ = 10; self.LastDeltaZ = 0; 
        # The deltaZ of ultra trap changes by DdeltaZ in each step starting from FirstDeltaZ (to LastDeltaZ).
                
    def dotask(self):
        self.traps = self.parent.pattern.pattern
        if isinstance(self.trap, QUltraTrap):
            fn0, fn_ext = os.path.splitext(self.parent.dvr.filename)
            self.register('Modify', group=self.traps, NewDeltaZ=self.FirstDeltaZ, NewDeltaPhi=0)
            for n in range(0, int(np.absolute((self.FirstDeltaZ-self.LastDeltaZ)/self.DdeltaZ))+1 ): 
                dZnew = np.absolute(self.FirstDeltaZ + self.DdeltaZ*n) 
                self.register('Delay', delay=50)
                if (dZnew>=0):
                    self.register('Record', fn=fn0+'{:03d}.avi'.format(int(np.abs(dZnew))),nframes=50)
                else:
                    self.register('Record', fn=fn0+'-{:03d}.avi'.format(int(np.abs(dZnew))),nframes=50)
                self.register('Delay', delay=10)
                self.register('Modify', group=self.traps, NewDeltaZ=dZnew, NewDeltaPhi=0) 
