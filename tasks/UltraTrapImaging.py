# -*- coding: utf-8 -*-
# MENU: Experiments/Ultra Trap Imaging

from .task import task
import numpy as np
from pyfablib.traps.QUltraTrap import QUltraTrap
import os
from pyqtgraph.Qt import QtGui



class UltraTrapImaging(task):
    """Delay, save an image, and move the ultra trap in the z direction."""

    def __init__(self, **kwargs):
        super(UltraTrapImaging, self).__init__(**kwargs)
        self.traps = None; self.trap = None;

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        for t in self.traps.flatten(): 
              if isinstance(t, QUltraTrap): self.trap = t; # selecting the ultra traps
        self.step = 1; self.start = -self.trap.deltaZ; self.end = self.trap.deltaZ; 

    def dotask(self):
        self.traps = self.parent.pattern.pattern
        if self.trap is not None:
            fn0, fn_ext = os.path.splitext(self.parent.dvr.filename)
            self.register('translate', traps=self.traps, dr=QtGui.QVector3D(0, 0, self.start-self.trap.r.z()))
            for n in range(0, int(np.absolute((self.end-self.start)/self.step))+1 ):
                self.register('delay', delay=5)
                self.register('MaxTaskWithName',fn='{:03d}'.format(int(self.start+n*self.step)))
                self.register('delay', delay=5)
                self.register('translate', traps=self.traps, dr=QtGui.QVector3D(0, 0, self.step))
