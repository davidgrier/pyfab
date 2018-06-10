# -*- coding: utf-8 -*-
# MENU: Set hologram

from task import task
import numpy as np


class vortex(task):
    def __init__(self, ell=10, **kwargs):
        super(vortex, self).__init__(**kwargs)
        self.ell = ell

    def dotask(self):
        cgh = self.parent.cgh
        theta = np.arctan2.outer(np.imag(cgh.iqx), np.imag(cgh.iqy))
        theta += np.pi
        phi = np.remainder(self.ell * theta, 2. * np.pi)
        cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))


class sethologram(task):
    """Set hologram"""

    def __init__(self, **kwargs):
        super(sethologram, self).__init__(**kwargs)
        self.kwargs = kwargs

    def dotask(self):
        self.register('cleartraps')
        self.register(vortex(**self.kwargs))
