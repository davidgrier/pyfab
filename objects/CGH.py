#!/usr/bin/env python

"""CGH.py: compute phase-only holograms for optical traps."""

import numpy as np
from PyQt4 import QtGui, QtCore
from numba import jit
import json
from time import time


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
        # Trap properties for current pattern
        self.trapdata = []

        # SLM geometry
        self.slm = slm
        self.w = self.slm.width()
        self.h = self.slm.height()

        # Conversion from SLM pixels to wavenumbers
        self._qpp = 2. * np.pi / self.w / 10.
        # Effective aspect ratio of SLM pixels
        self._alpha = 1.
        # Location of optical axis in SLM coordinates
        self._rs = QtCore.QPointF(self.w / 2., self.h / 2.)
        self.updateGeometry()

        # Coordinate transformation matrix for trap locations
        self.m = QtGui.QMatrix4x4()
        # Location of optical axis in camera coordinates
        self._rc = QtGui.QVector3D(320., 240., 0.)
        # Orientation of camera relative to SLM
        self._theta = 0.
        self.updateTransformationMatrix()

    @jit(parallel=True)
    def quantize(self):
        phi = ((128. / np.pi) * np.angle(self._psi) + 127.).astype(np.uint8)
        return phi.T

    @jit(parallel=True)
    def compute_one(self, amp, r):
        """Compute phase hologram to displace a trap with
        a specified complex amplitude to a specified position
        """
        ex = np.exp(self.iqx * r.x() + self.iqxsq * r.z())
        ey = np.exp(self.iqy * r.y() + self.iqysq * r.z())
        self._psi += np.outer(amp * ex, ey, self._delta)

    def window(self, r):
        x = [r.x() / self.w, r.y() / self.h]
        fac = 1. / np.prod(np.sinc(x))
        return np.min((fac * fac, 100.))

    @jit(parallel=True)
    def compute(self):
        """Compute phase hologram for specified traps
        """
        start = time()
        self._psi.fill(0. + 0j)
        for properties in self.trapdata:
            r = self.m * properties['r']
            amp = properties['amp'] * self.window(r)
            self.compute_one(amp, r)
        self.slm.data = self.quantize()
        self.time = time() - start

    def outertheta(self, x, y):
        return np.arctan2.outer(y, x)

    def updateGeometry(self):
        """Compute position-dependent properties in SLM plane
        and allocate buffers.
        """
        shape = (self.w, self.h)
        self._psi = np.zeros(shape, dtype=np.complex_)
        self._delta
        qx = np.arange(self.w) - self.rs.x()
        qy = np.arange(self.h) - self.rs.y()
        qx = self.qpp * qx
        qy = self.alpha * self.qpp * qy
        self.iqx = 1j * qx
        self.iqy = 1j * qy
        self.iqxsq = 1j * qx * qx
        self.iqysq = 1j * qy * qy
        self.itheta = 1j * self.outertheta(qx, qy)

    @property
    def rs(self):
        return self._rs

    @rs.setter
    def rs(self, rs):
        if isinstance(rs, QtCore.QPointF):
            self._rs = rs
        else:
            self._rs = QtCore.QPointF(rs[0], rs[1])
        self.updateGeometry()
        self.compute()

    @property
    def qpp(self):
        return self._qpp

    @qpp.setter
    def qpp(self, qpp):
        self._qpp = float(qpp)
        self.updateGeometry()
        self.compute()

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)
        self.updateGeometry()
        self.compute()

    def updateTransformationMatrix(self):
        self.m.setToIdentity()
        self.m.rotate(self.theta, 0., 0., 1.)
        self.m.translate(-self.rc)

    @property
    def rc(self):
        return self._rc

    @rc.setter
    def rc(self, rc):
        if isinstance(rc, QtGui.QVector3D):
            self._rc = rc
        else:
            self._rc = QtGui.QVector3D(rc[0], rc[1], rc[2])
        self.updateTransformationMatrix()
        self.compute()

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = float(theta)
        self.updateTransformationMatrix()
        self.compute()

    def setData(self, trapdata):
        self.trapdata = trapdata
        self.compute()

    @property
    def calibration(self):
        return {'qpp': self.qpp,
                'alpha': self.alpha,
                'rs': (self.rs.x(), self.rs.y()),
                'rc': (self.rc.x(), self.rc.y(), self.rc.z()),
                'theta': self.theta}

    @calibration.setter
    def calibration(self, values):
        if not isinstance(values, dict):
            return
        for attribute, value in values.iteritems():
            try:
                setattr(self, attribute, value)
            except AttributeError:
                print('unknown attribute:', attribute)

    def serialize(self):
        return json.dumps(self.calibration,
                          indent=2,
                          separators=(',', ': '),
                          ensure_ascii=False)

    def deserialize(self, s):
        values = json.loads(s)
        self.calibration = values
