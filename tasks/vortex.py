# -*- coding: utf-8 -*-
# MENU: Vortex trap

from .sethologram import sethologram
import numpy as np


class vortex(sethologram):
    """Set vortex hologram"""
    
    def __init__(self, ell=10, **kwargs):
        super(vortex, self).__init__(**kwargs)
        self.ell = ell

    def dotask(self):
        theta = np.arctan2.outer(self.qx, self.qy)
        theta += np.pi
        phi = np.remainder(self.ell * theta, 2. * np.pi)
        self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))
