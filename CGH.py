#!/usr/bin/env python

"""CGH.py: compute phase-only holograms for optical traps."""

import numpy as np
from PyQt4 import QtGui, QtCore
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

        # SLM geometry
        self.slm = slm
        self.w = self.slm.width()
        self.h = self.slm.height()
        # Conversion from SLM pixels to wavenumbers
        # Calibration constant:
        # qpp: float
        self._qpp = 2. * np.pi / self.w / 10.
        # Effective aspect ratio of SLM pixels
        # Calibration constant:
        # alpha: float
        self._alpha = 1.
        # Location of optical axis in SLM coordinates
        # Calibration constant:
        # rs: QPointF
        self.rs = QtCore.QPointF(self.w / 2., self.h / 2.)

        # Coordinate transformation matrix for trap locations
        self.m = QtGui.QMatrix4x4()
        self._rc = QtCore.QPointF()
        self._theta = 0.
        # Location of optical axis in camera coordinates
        # Calibration constant:
        # rc: QPointF
        self.rc = QtCore.QPointF(320., 240.)
        # Orientation of camera relative to SLM
        # Calibration constant:
        # theta: float
        self.theta = 0.

    def updateGeometry(self):
        """Compute position-dependent properties in SLM plane.
        """
        qx = np.linspace(-self.rs.x(), self.w - 1 - self.rs.x(), self.w)
        qy = np.linspace(-self.rs.y(), self.h - 1 - self.rs.y(), self.h)
        qx = self._qpp * qx
        qy = self._alpha * self._qpp * qy
        self.iqx = 1j * qx
        self.iqy = 1j * qy
        self.iqxsq = 1j * qx * qx
        self.iqysq = 1j * qy * qy

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, rs):
        self._rs = rs
        self.updateGeometry()

    @property
    def qpp(self):
        return self._qpp

    @qpp.setter
    def qpp(self, value):
        self._qpp = value
        self.updateGeometry()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value
        self.updateGeometry()

    def updateTransformationMatrix(self):
        self.m.setToIdentity()
        self.m.translate(-self._rc.x(), -self._rc.y())
        self.m.rotate(self._theta, 0., 0., 1.)

    @property
    def rc(self):
        return self._rc

    @rc.setter
    def rc(self, value):
        self._rc = value
        self.updateTransformationMatrix()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._theta = value
        self.updateTransformationMatrix()

    @jit
    def setData(self, properties):
        psi = np.zeros((self.w, self.h), dtype=np.complex_)
        for property in properties:
            r = self.m * property['r']
            amp = property['a'] * np.exp(1j * property['phi'])
            ex = np.exp(self.iqx * r.x() + self.iqxsq * r.z())
            ey = np.exp(self.iqy * r.y() + self.iqysq * r.z())
            psi += np.outer(amp * ey, ex)
        phi = (256. * (np.angle(psi) / np.pi + 1.)).astype(np.uint8)
        self.slm.setData(phi)
