#!/usr/bin/env python

"""CGH.py: compute phase-only holograms for optical traps."""

import numpy as np
from PyQt4 import QtGui
from numba import jit


class CGH(object):

    def __init__(self, slm=None):
        self.slm = slm
        self.w = slm.width()
        self.h = slm.height()
        sx, sy = slm.center
        self.rc = QtGui.QVector3D(320., 240., 0)
        factor = 2. * np.pi / self.w / 10.
        qx = np.linspace(0 - sx, self.w - 1 - sx, self.w)
        qy = np.linspace(0 - sy, self.h - 1 - sy, self.h)
        self.iqx = factor * 1j * qx
        self.iqy = factor * 1j * qy

    @jit
    def setData(self, properties):
        psi = np.zeros((self.w, self.h), dtype=np.complex_)
        for property in properties:
            r = property['r'] - self.rc
            fac = property['a'] * np.exp(1j * property['phi'])
            psi += np.outer(fac * np.exp(self.iqy * r.y()),
                            np.exp(self.iqx * r.x()))
        phi = (256. * (np.angle(psi) / np.pi + 1.)).astype(np.uint8)
        self.slm.setData(phi)
