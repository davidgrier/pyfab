# coding: utf-8
# MENU: Bessel

from .sethologram import sethologram
import numpy as np


class bessel(sethologram):
    """Set hologram for Bessel trap."""

    def __init__(self, **kwargs):
        super(bessel, self).__init__(**kwargs)


    def bessel(self, qx, qy, shift=100, shift0=0, Modetot=1):
        shift=shift
        shift0=shift0
        Modetot=Modetot
        qr = np.hypot.outer(qx, qy)
        phi = np.remainder(qr, 2*np.pi)
        return phi.T
        
    def dotask(self):
        #xv, yv = np.meshgrid(self.qx, self.qy)
        #phi = np.zeros(shape=(self.qx.shape[0], self.qy.shape[0]))
        #phi.fill(200.)
        phi = self.bessel(self.qx, self.qy)
        self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))
