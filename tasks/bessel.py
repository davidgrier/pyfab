# coding: utf-8
# MENU: Bessel

from .sethologram import sethologram
import numpy as np


class bessel(sethologram):
    """Set hologram for Bessel trap."""

    def __init__(self, **kwargs):
        super(bessel, self).__init__(**kwargs)
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