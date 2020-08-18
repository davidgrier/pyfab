# coding: utf-8
# MENU: Set background/Bessel beam

from .SetHologram import SetHologram
import numpy as np


class Bessel(SetHologram):
    """Set hologram for Bessel trap."""

    def __init__(self, **kwargs):
        super(Bessel, self).__init__(**kwargs)
        self.shift = 100
        self.shiftx = 0
        self.shifty = 0
        self.modetot = 1

    def dotask(self):
        xv, yv = np.meshgrid(self.qx, self.qy)
        xv, yv = xv.T, yv.T
        qr = np.hypot.outer(self.qx, self.qy)
        phi = np.remainder(np.angle(self.modetot) - self.shift*qr -
                           self.shiftx*xv - self.shifty*yv, 2*(np.pi))
        self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))
