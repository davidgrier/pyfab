# -*- coding: utf-8 -*-

'''Demonstration of autotrapping then iteratively delaying, recording video,
 and translating traps in the z direction.'''

from autotrap import autotrap
from pyqtgraph.Qt import QtGui


class moverecordz(autotrap):
    
    def __init__(self, **kwargs):
        super(moverecordz, self).__init__(**kwargs)

    def dotask(self):
        if self.traps is not None:
            register = self.parent.tasks.registerTask
            fn = self.parent.dvr.filename
            fn = fn[0:-4]
            z = self.traps.r.z()
            dz = 1
            dr = QtGui.QVector3D(0, 0, dz)
            for n in range(0, 5):
                register('delay', delay=50)
                register('record', nframes=100, fn=fn + str(z + n) + '.avi')
                register('translate', traps=self.traps, dr=dr)


