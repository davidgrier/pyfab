#!/usr/bin/env python

"""CGH.py: compute phase-only holograms for optical traps."""

import numpy as np
from PyQt4 import QtGui
from numba import jit


class CGH(object):
    """Base class for computing computer-generated holograms.

    For each trap, the coordinate r obtained from the fabscreen
    is measured relative to the calibrated location rc of the
    zeroth-order focal point, which itself is measured relative to
    the center of the focal plane. The resulting displacement is
    projected onto the coordinate system in the SLM place.
    Projection involves a calibrated rotation about z with
    a rotation matrix m.

    The hologram is computed using calibrated wavenumbers for
    the Cartesian coordinates in the SLM plane.  These differ from
    each other because the SLM is likely to be tilted relative to the
    optical axis.
    """

    def __init__(self, slm=None):

        self.slm = slm
        # geometry in SLM plane
        self.precompute()
        # geometry in video plane
        self.rc = QtGui.QVector3D(320., 240., 0)
        self.m = QtGui.QMatrix4x4()
        self.m.setToIdentity()

    def precompute(self):
        sx, sy = self.slm.center
        self.w = self.slm.width()
        self.h = self.slm.height()
        factor = 2. * np.pi / self.w / 10.
        qx = factor * np.linspace(0 - sx, self.w - 1 - sx, self.w)
        qy = factor * np.linspace(0 - sy, self.h - 1 - sy, self.h)
        self.iqx = 1j * qx
        self.iqy = 1j * qy
        self.iqxsq = 1j * qx * qx
        self.iqysq = 1j * qy * qy

    @jit
    def setData(self, properties):
        psi = np.zeros((self.w, self.h), dtype=np.complex_)
        for property in properties:
            r = self.m * (property['r'] - self.rc)
            amp = property['a'] * np.exp(1j * property['phi'])
            ex = np.exp(self.iqx * r.x() + self.iqxsq * r.z())
            ey = np.exp(self.iqy * r.y() + self.iqysq * r.z())
            psi += np.outer(amp* ey, ex)
        phi = (256. * (np.angle(psi) / np.pi + 1.)).astype(np.uint8)
        self.slm.setData(phi)
