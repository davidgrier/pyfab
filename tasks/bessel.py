# coding: utf-8
# MENU: Bessel

from .sethologram import sethologram
import numpy as np


class bessel(sethologram):
    """Set hologram for Bessel trap."""

    def __init__(self, **kwargs):
        super(bessel, self).__init__(**kwargs)


    def bessel(self, x, y, shift=100, shift0=0, Modetot=1):
        shift=shift
        shift0=shift0
        Modetot=Modetot
        phi = np.remainder(np.angle(Modetot)-shift*(np.sqrt(x**2+y**2))-shift0*(x),2*(np.pi))
        print(phi)
        return phi.T
        
    def dotask(self):
        xv, yv = np.meshgrid(self.qx, self.qy)
        phi = self.bessel(xv, yv)
        print(((255./(2.*np.pi))*phi).astype(np.uint8))
        self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))
