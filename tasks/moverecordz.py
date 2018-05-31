# -*- coding: utf-8 -*-

'''Demonstration of autotrapping then iteratively delaying, recording video,
 and translating traps in the z direction.'''

from autotrap import autotrap


class moverecordz(autotrap):
    
    def __init__(self, **kwargs):
        super(moverecordz, self).__init__(**kwargs)

    def dotask(self):
        if self.traps is not None:
            register = self.parent.tasks.registerTask
            fn = self.parent.dvr.filename
            fn = fn[0:-4]
            z = self.traps.r.z()
            #wstage = self.parent.wstage
            #if wstage is not None:
            #    z = wstage.instrument.z()
            for n in range(1, 5):
                register('delay')
                register('record', fn=fn + str(z + n) + '.avi')
                register('translatez', traps=self.traps)


