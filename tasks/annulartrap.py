# -*- coding: utf-8 -*-
# MENU: Ring trap

from .sethologram import sethologram
import numpy as np
from scipy.special import jv
from PyQt4.QtGui import QInputDialog
import logging


class annulartrap(sethologram):
    """Set hologram for ring trap."""
    def __init__(self, **kwargs):
        super(annulartrap, self).__init__(**kwargs)
        qtext, self.ok = QInputDialog.getText(self.parent,
                                              'Set parameters',
                                              'R, m, dx, dy, dz:')
        self.params = qtext.split(',')
        R, m, dx, dy, dz = 300, 40, 0, 0, 0
        if self.ok:
            if len(self.params) == 5:
                try:
                    R, m, dx, dy, dz = map(lambda param: float(param),
                                           self.params)                
                except Exception as e:
                    logging.error('Could not set parameters: {}'.format(e))
        self.R = R
        self.m = m
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def dotask(self):
        if self.ok:
            # Set coordinates
            rho = np.hypot.outer(self.qx, self.qy)
            phi = np.arctan2.outer(self.qx, self.qy)
            xv, yv = np.meshgrid(self.qx, self.qy)
            xv, yv = xv.T, yv.T
            # Calculate phase
            psi = jv(self.m, self.R * rho) * np.exp(1j * self.m * phi)
            phi = np.array(np.angle(psi) + np.pi)
            phi -= np.amin(phi)
            phi_disp = np.array(self.dx*xv + self.dy*yv + (rho**2)*self.dz)
            phi_disp = phi_disp - np.amin(phi_disp)
            phi_disp *= 2*np.pi
            phi += phi_disp
            phi = phi % (2*np.pi)
            # Set phase
            self.cgh.setPhi(((255./(2.*np.pi))*phi).astype(np.uint8))
