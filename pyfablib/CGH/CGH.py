# -*- coding: utf-8 -*-

"""CGH.py: compute phase-only holograms for optical traps."""

import numpy as np
from PyQt4 import QtGui, QtCore
from numba import jit
from time import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class CGH(QtCore.QObject):
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

    NOTE: This version has thread-safe slots for setting parameters
    (setProperty) and for triggering computations (setTraps).
    It emits a thread-safe signal (sigHologramReady) to transfer
    computed holograms.
    """

    sigRun = QtCore.pyqtSignal(bool)
    sigComputing = QtCore.pyqtSignal(bool)
    sigHologramReady = QtCore.pyqtSignal(np.ndarray)
    sigUpdateGeometry = QtCore.pyqtSignal()

    def __init__(self, slm=None):
        super(CGH, self).__init__()
        self.traps = []
        # SLM geometry
        self.slm = slm
        self.w = self.slm.width()
        self.h = self.slm.height()
        self.shape = (self.w, self.h)
        self.phi = np.zeros(self.shape).astype(np.uint8)

        # Conversion from SLM pixels to wavenumbers
        self._qpp = 2. * np.pi / self.w / 10.
        # Effective aspect ratio of SLM pixels
        self._alpha = 1.
        # Location of optical axis in SLM coordinates
        self._rs = QtCore.QPointF(self.w / 2., self.h / 2.)

        # Coordinate transformation matrix for trap locations
        self.m = QtGui.QMatrix4x4()
        # Location of optical axis in camera coordinates
        self._rc = QtGui.QVector3D(320., 240., 0.)
        # Orientation of camera relative to SLM
        self._thetac = 0.
        # Splay wavenumber
        self._k0 = 0.01

    @QtCore.pyqtSlot()
    def start(self):
        logger.info('starting CGH pipeline')
        self.updateGeometry()
        self.updateTransformationMatrix()

    @QtCore.pyqtSlot()
    def stop(self):
        logger.info('stopping CGH pipeline')

    @QtCore.pyqtSlot(object, object)
    def setProperty(self, name, value):
        setattr(self, name, value)

    @QtCore.pyqtSlot(object)
    def setTraps(self, traps):
        self.traps = traps
        self.compute()

    @jit(parallel=True)
    def quantize(self, psi):
        self.phi = ((128. / np.pi) * np.angle(psi) + 127.).astype(np.uint8)
        return self.phi.T

    @jit(parallel=True)
    def compute_displace(self, amp, r, buffer):
        """Compute phase hologram to displace a trap with
        a specified complex amplitude to a specified position
        """
        ex = np.exp(self.iqx * r.x() + self.iqxsq * r.z())
        ey = np.exp(self.iqy * r.y() + self.iqysq * r.z())
        np.outer(amp * ex, ey, buffer)

    def window(self, r):
        x = 0.5 * np.pi * np.array([r.x() / self.w, r.y() / self.h])
        fac = 1. / np.prod(np.sinc(x))
        return np.min((np.abs(fac), 100.))

    @jit(parallel=True)
    def compute(self, all=False):
        """Compute phase hologram for specified traps
        """
        self.sigComputing.emit(True)
        start = time()
        self._psi.fill(0. + 0j)
        for trap in self.traps:
            if ((all is True) or trap.needsUpdate):
                # map coordinates into trap space
                r = self.m * trap.r
                # axial splay
                fac = 1. / (1. + self.k0 * (r.z() - self.rc.z()))
                r *= QtGui.QVector3D(fac, fac, 1.)
                # windowing
                amp = trap.amp * self.window(r)
                amp = trap.amp
                if trap.psi is None:
                    trap.psi = self._psi.copy()
                self.compute_displace(amp, r, trap.psi)
                trap.needsUpdate = False
            self._psi += trap.structure * trap.psi
        self.sigHologramReady.emit(self.quantize(self._psi))
        self.time = time() - start
        self.sigComputing.emit(False)

    def bless(self, field):
        if type(field) is complex:
            field = np.ones(self.shape)
        return field.astype(np.complex_)

    def updateGeometry(self):
        """Compute position-dependent properties in SLM plane
        and allocate buffers.
        """
        self._psi = np.zeros(self.shape, dtype=np.complex_)
        qx = np.arange(self.w) - self.rs.x()
        qy = np.arange(self.h) - self.rs.y()
        qx = self._qpp * qx
        qy = self._alpha * self._qpp * qy
        self.iqx = 1j * qx
        self.iqy = 1j * qy
        self.iqxsq = 1j * qx * qx
        self.iqysq = 1j * qy * qy
        self.theta = np.arctan2.outer(qx, qy)
        self.qr = np.hypot.outer(qx, qy)
        self.sigUpdateGeometry.emit()

    @property
    def xs(self):
        return self.rs.x()

    @xs.setter
    def xs(self, xs):
        rs = self.rs
        rs.setX(xs)
        self.rs = rs

    @property
    def ys(self):
        return self.rs.y()

    @ys.setter
    def ys(self, ys):
        rs = self.rs
        rs.setY(ys)
        self.rs = rs

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
        self.compute(all=True)

    @property
    def qpp(self):
        return self._qpp * 1000.

    @qpp.setter
    def qpp(self, qpp):
        self._qpp = float(qpp) / 1000.
        self.updateGeometry()
        self.compute(all=True)

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = float(alpha)
        self.updateGeometry()
        self.compute(all=True)

    def updateTransformationMatrix(self):
        self.m.setToIdentity()
        self.m.rotate(self.thetac, 0., 0., 1.)
        self.m.translate(-self.rc)

    @property
    def xc(self):
        return self.rc.x()

    @xc.setter
    def xc(self, xc):
        rc = self.rc
        rc.setX(xc)
        self.rc = rc

    @property
    def yc(self):
        return self.rc.y()

    @yc.setter
    def yc(self, yc):
        rc = self.rc
        rc.setY(yc)
        self.rc = rc

    @property
    def zc(self):
        return self.rc.z()

    @zc.setter
    def zc(self, zc):
        rc = self.rc
        rc.setZ(zc)
        self.rc = rc

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
        self.compute(all=True)

    @property
    def thetac(self):
        return self._thetac

    @thetac.setter
    def thetac(self, thetac):
        self._thetac = float(thetac)
        self.updateTransformationMatrix()
        self.compute(all=True)

    @property
    def k0(self):
        return self._k0

    @k0.setter
    def k0(self, k0):
        self._k0 = float(k0)
        self.compute(all=True)

    def setPhi(self, phi):
        self.phi = phi.astype(np.uint8)
        self.sigHologramReady.emit(self.phi.T)
