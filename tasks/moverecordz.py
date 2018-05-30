# -*- coding: utf-8 -*-

'''Demonstration of autotrapping then iteratively delaying, recording video, and translating traps in the z direction.'''

from autotrap import autotrap
from pyqtgraph.Qt import QtGui


class moverecordz(autotrap):
    
    def __init__(self, **kwargs):
        super(moverecordz, self).__init__(**kwargs)

    def dotask(self):
        if self.traps is not None:
            dz = 1
            dr = QtGui.QVector3D(0, 0, dz)
            register = self.parent.tasks.registerTask
            z = self.traps.r.z()
            #wstage = self.parent.wstage
            #if wstage is not None:
            #    z = wstage.instrument.z()
            for n in range(1, 5):
                register('delay')
                register('record', fn='~/data/moveRecordZ' + str(z + n*dz) + '.avi')
                self.traps.select(True)
                self.traps.moveBy(dr)
                self.traps.select(False)

