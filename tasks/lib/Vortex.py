# -*- coding: utf-8 -*-
"""# #MENU: Set background/Optical vortex"""

from .SetHologram import SetHologram
import numpy as np

## is setHologram outdated?
class Vortex(SetHologram):
    """Set vortex hologram"""

    def __init__(self, ell=10, **kwargs):
        super(Vortex, self).__init__(**kwargs)
        self.ell = ell

    def dotask(self):
        theta = np.arctan2.outer(self.qx, self.qy)
        theta += np.pi
        phi = np.remainder(self.ell * theta, 2. * np.pi)
        print(((255./(2.*np.pi))*phi).astype(np.uint8))
        self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))
