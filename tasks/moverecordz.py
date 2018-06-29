# -*- coding: utf-8 -*-
# MENU: Measure z

from .task import task
from pyqtgraph.Qt import QtGui
import numpy as np


class moverecordz(task):
    """Delay, record, and translate traps in the z direction."""
    
    def __init__(self, **kwargs):
        super(moverecordz, self).__init__(**kwargs)
        self.traps = None

    def dotask(self):
        self.traps = self.parent.pattern.pattern
        if self.traps.count() > 0:
            fn = self.parent.dvr.filename[0:-4]
            register = self.parent.tasks.registerTask
            z = self.traps.r.z()
            dz = 20
            dr = QtGui.QVector3D(0, 0, dz)
            for n in range(0, 50):
                register('delay', delay=100)
                register('record', nframes=100, fn=fn + '{:03d}.avi'.format(str(np.absolute(z + dz*n)))
                register('translate', traps=self.traps, dr=dr)
