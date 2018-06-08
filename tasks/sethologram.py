# -*- coding: utf-8 -*-
# MENU: Set hologram

from task import task
import numpy as np


class hologram(task):

    def __init__(self, **kwargs):
        super(hologram, self).__init__(**kwargs)

    def initialize(self, frame):
        ell = 10
        cgh = self.parent.cgh
        theta = np.arctan2.outer(np.imag(cgh.iqx), np.imag(cgh.iqy))
        theta += np.pi
        phi = np.remainder(ell * theta, 2. * np.pi)
        cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))


class sethologram(task):
    """Set hologram"""

    def __init__(self, **kwargs):
        super(sethologram, self).__init__(**kwargs)

    def initialize(self, frame):
        register = self.parent.tasks.registerTask
        register('cleartraps')
        register(hologram())
