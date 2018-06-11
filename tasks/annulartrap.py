# -*- coding: utf-8 -*-
# MENU: Ring trap

from .sethologram import sethologram
import numpy as np
from scipy.special import jv


class annulartrap(sethologram):
    """Set hologram for ring trap."""
    
    def __init__(self, **kwargs):
        super(annulartrap, self).__init__(**kwargs)

    def dotask(self):
        xv, yv = np.meshgrid(self.qx, self.qy)
        phi = self.annulartrap(xv, yv)
        self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))

    def annulartrap(self, x, y, dx=0., dy=0., dz=0.):
        R = 1000.                         # Radius of the ring [pixels]
        m = 9.                          # m-fold phase modulation around ring
        lamda = 1.064                   # wavelength [pixels]
        f = 100.                        # focal length [pixels]
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        k = 2 * np.pi * R / (lamda * f)
        psi = jv(m, k * rho) * np.exp(1j * m * phi)
        phi = np.array(np.angle(psi) + np.pi)
        phi -= np.amin(phi)
        phi_disp = np.array(dx*x+dy*y+(x**2+y**2)*dz)
        phi_disp = phi_disp - np.amin(phi_disp)
        phi_disp *= 2*np.pi
        # Add the phases together
        phi += phi_disp
        phi = phi % (2*np.pi)
        return phi.T
