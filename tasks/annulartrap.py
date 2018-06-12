# -*- coding: utf-8 -*-
# MENU: Ring trap

from .sethologram import sethologram
import numpy as np
from scipy.special import jv
from PyQt4.QtGui import QInputDialog


class annulartrap(sethologram):
    """Set hologram for ring trap."""
    
    def __init__(self, **kwargs):
        super(annulartrap, self).__init__(**kwargs)
        qtext, self.ok = QInputDialog.getText(self.parent,
                                         'Set parameters',
                                         'R, m, dx, dy, dz:')
        self.params = qtext.split(',')

    def dotask(self):
        if self.ok:
            xv, yv = np.meshgrid(self.qx, self.qy)
            if len(self.params) == 5:
                try:
                    R, m, dx, dy, dz = map(lambda param: float(param), self.params)
                    phi = self.annulartrap(xv, yv,
                                           R=R, m=m, dx=dx, dy=dy, dz=dz)
                except Exception as e:
                    print(e)
                    print('Enter numerical values for R, m, dx, dy, dz separated by commas.')
                    return
            else:
                print('Custom parameters not found. Falling back to default.')
                phi = self.annulartrap(xv, yv)
            self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))

    def annulartrap(self, x, y, R=50., m=9., dx=0., dy=0., dz=0.):
        R = R                         # Radius of the ring [pixels]
        m = m                          # m-fold phase modulation around ring
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        psi = jv(m, R * rho) * np.exp(1j * m * phi)
        phi = np.array(np.angle(psi) + np.pi)
        phi -= np.amin(phi)
        phi_disp = np.array(dx*x+dy*y+(x**2+y**2)*dz)
        phi_disp = phi_disp - np.amin(phi_disp)
        phi_disp *= 2*np.pi
        # Add the phases together
        phi += phi_disp
        phi = phi % (2*np.pi)
        return phi.T
