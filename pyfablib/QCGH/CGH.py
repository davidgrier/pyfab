# -*- coding: utf-8 -*-

"""CGH.py: compute phase-only holograms for optical traps."""

import numpy as np
from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot, QPointF)
from PyQt5.QtGui import (QVector3D, QMatrix4x4)
from numba import jit
from time import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class CGH(QObject):
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

    sigRun = pyqtSignal(bool)
    sigComputing = pyqtSignal(bool)
    sigHologramReady = pyqtSignal(np.ndarray)
    sigUpdateGeometry = pyqtSignal()
    sigUpdateTransformationMatrix = pyqtSignal()

    def __init__(self, slm=None):
        super(CGH, self).__init__()
        self.traps = []
        # SLM geometry
        self.slm = slm
        self.h = self.slm.height()
        self.w = self.slm.width()
        self.shape = (self.h, self.w)
        self.phi = np.zeros(self.shape).astype(np.uint8)

        # Instrument properties
        # vacuum wavelength of trapping laser [um]
        self._wavelength = 1.064
        # refractive index of medium
        self._refractiveIndex = 1.340
        # magnification of objective lens
        self._magnification = 100.
        # focal length of objective lens [um]
        self._focalLength = 200.
        # camera pitch [um/pixel]
        self._cameraPitch = 4.8
        # SLM pitch [um/pixel]
        self._slmPitch = 8.
        # SLM scale factor
        self._scaleFactor = 3.

        # Coordinate transformation matrix for trap locations
        self._m = QMatrix4x4()
        # Location of optical axis in camera coordinates
        self._rc = QVector3D(320., 240., 0.)
        # Orientation of camera relative to SLM [degrees]
        self._thetac = 0.

        # Location of optical axis in SLM coordinates
        self._rs = QPointF(self.w / 2., self.h / 2.)
        # Tilt of SLM relative to optical axis [degrees]
        self._phis = 8.

        # Computed calibration constants
        # Conversion from SLM pixels to wavenumbers
        self._qpp = 2. * np.pi / self.w / 10.
        # Effective aspect ratio of SLM pixels
        self._alpha = 1.
        # Effective axial aspect ratio: lambda/4 [pixel]
        self._beta = 2.
        # Splay wavenumber
        self._k0 = 0.01

    # Slots for threaded operation
    @pyqtSlot()
    def start(self):
        logger.info('starting CGH pipeline')
        self.updateGeometry()
        self.updateTransformationMatrix()

    @pyqtSlot()
    def stop(self):
        logger.info('stopping CGH pipeline')

    @pyqtSlot(object, object)
    def setProperty(self, name, value):
        setattr(self, name, value)

    @pyqtSlot(object)
    def setTraps(self, traps):
        self.traps = traps
        self.compute()

    # Methods for computing holograms
    @staticmethod
    @jit(nopython=True)
    def quantize(psi):
        """Compute the phase of the field, scaled to uint8"""
        return ((128. / np.pi) * np.angle(psi) + 127.).astype(np.uint8)

    def window(self, r):
        """Adjust amplitude to account for aperture size"""
        x = 0.5 * np.pi * np.array([r.x() / self.w, r.y() / self.h])
        fac = 1. / np.prod(np.sinc(x))
        return np.min((np.abs(fac), 100.))

    # @jit
    def compute_displace(self, amp, r, buffer):
        """Compute phase hologram to displace a trap with
        a specified complex amplitude to a specified position
        """
        ex = np.exp(self.iqx * r.x() + self.iqxsq * r.z())
        ey = np.exp(self.iqy * r.y() + self.iqysq * r.z())
        np.outer(amp * ey, ex, buffer)

    # @jit
    def compute(self, all=False):
        """Compute phase hologram for specified traps"""
        self.sigComputing.emit(True)
        start = time()
        self._psi.fill(0. + 0j)
        for trap in self.traps:
            if ((all is True) or trap.needsRefresh):
                # map coordinates into trap space
                r = self.m * trap.r
                # axial splay
                fac = 1. / (1. + self.k0 * (r.z() - self.rc.z()))
                r *= QVector3D(fac, fac, 1.)
                # windowing
                # amp = trap.amp * self.window(r)
                amp = trap.amp
                if trap.psi is None:
                    trap.psi = self._psi.copy()
                self.compute_displace(amp, r, trap.psi)
                trap.needsUpdate = False
            self._psi += trap.structure * trap.psi
        self.phi = self.quantize(self._psi)
        self.sigHologramReady.emit(self.phi)
        self.time = time() - start
        self.sigComputing.emit(False)

    def bless(self, field):
        """Ensure that field has correct type for compute"""
        if type(field) is complex:
            field = np.ones(self.shape)
        return field.astype(np.complex_)

    def setPhi(self, phi):
        """Specify the hologram to project, without computation"""
        self.phi = phi.astype(np.uint8)
        self.sigHologramReady.emit(self.phi.T)

    # Helper routines when calibration constants are changed
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
        self.iqxsq = 1j * self._beta * qx * qx
        self.iqysq = 1j * self._beta * qy * qy
        self.theta = np.arctan2.outer(qy, qx)
        self.qr = np.hypot.outer(qy, qx)
        self.sigUpdateGeometry.emit()

    def updateTransformationMatrix(self):
        """Translate and rotate requested trap positions to account
        for position and orientation of camera relative to SLM
        """
        self.m.setToIdentity()
        self.m.rotate(self.thetac, 0., 0., 1.)
        self.m.translate(-self.rc)
        self.sigUpdateTransformationMatrix.emit()

    # Calibration constants
    # 1. Instrument parameters
    @property
    def wavelength(self):
        '''Vacuum wavelength of trapping laser'''
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = wavelength
        self.updateGeometry()
        self.compute(all=True)

    @property
    def refractiveIndex(self):
        '''Refractive index of medium'''
        return self._refractiveIndex

    @refractiveIndex.setter
    def refractiveIndex(self, refractiveIndex):
        self._refractiveIndex = refractiveIndex
        self.updateGeometry
        self.compute(all=True)

    @property
    def magnification(self):
        '''Magnification of objective lens'''
        return self._magnification

    @magnification.setter
    def magnification(self, magnification):
        self._magnification = magnification
        self.updateGeometry()
        self.compute(all=True)

    @property
    def focalLength(self):
        '''Focal length of objective lens [um]'''
        return self._focalLength

    @focalLength.setter
    def focalLength(self, focalLength):
        self._focalLength = focalLength
        self.updateGeometry()
        self.compute(all=True)

    @property
    def cameraPitch(self):
        '''Pixel pitch of camera [um/pixel]'''
        return self._cameraPitch

    @cameraPitch.setter
    def cameraPitch(self, cameraPitch):
        self._cameraPitch = cameraPitch
        self.updateGeometry()
        self.compute(all=True)

    @property
    def slmPitch(self):
        '''Pixel pitch of SLM [um/pixel]'''
        return self._slmPitch

    @slmPitch.setter
    def slmPitch(self, slmPitch):
        self._slmPitch = slmPitch
        self.updateGeometry()
        self.compute(all=True)

    @property
    def scaleFactor(self):
        '''SLM scale factor'''
        return self._scaleFactor

    @scaleFactor.setter
    def scaleFactor(self, scaleFactor):
        self._scaleFactor = scaleFactor
        self.updateGeometry()
        self.compute(all=True)

    # Calibration constants
    # 2. Camera plane
    @property
    def xc(self):
        '''x coordinate of optical axis in camera plane [pixels]'''
        return self.rc.x()

    @xc.setter
    def xc(self, xc):
        rc = self.rc
        rc.setX(xc)
        self.rc = rc

    @property
    def yc(self):
        '''y coordinate of optical axis in camera plane [pixels]'''
        return self.rc.y()

    @yc.setter
    def yc(self, yc):
        rc = self.rc
        rc.setY(yc)
        self.rc = rc

    @property
    def zc(self):
        '''Axial displacement of trapping plane
        from camera plane [pixels]'''
        return self.rc.z()

    @zc.setter
    def zc(self, zc):
        rc = self.rc
        rc.setZ(zc)
        self.rc = rc

    @property
    def rc(self):
        '''Location of optical axis in camera plane [pixels]'''
        return self._rc

    @rc.setter
    def rc(self, rc):
        if isinstance(rc, QVector3D):
            self._rc = rc
        else:
            self._rc = QVector3D(rc[0], rc[1], rc[2])
        self.updateTransformationMatrix()
        self.compute(all=True)

    @property
    def thetac(self):
        '''Orientation of camera relative to SLM [degrees]'''
        return self._thetac

    @thetac.setter
    def thetac(self, thetac):
        self._thetac = float(thetac)
        self.updateTransformationMatrix()
        self.compute(all=True)

    # Calibration constants
    # 3. SLM plane
    @property
    def xs(self):
        '''x coordinate of optical axis on SLM [pixels]'''
        return self.rs.x()

    @xs.setter
    def xs(self, xs):
        rs = self.rs
        rs.setX(xs)
        self.rs = rs

    @property
    def ys(self):
        '''y coordinate of optical axis on SLM [pixels]'''
        return self.rs.y()

    @ys.setter
    def ys(self, ys):
        rs = self.rs
        rs.setY(ys)
        self.rs = rs

    @property
    def rs(self):
        '''Location of optical axis on SLM [pixels]'''
        return self._rs

    @rs.setter
    def rs(self, rs):
        if isinstance(rs, QPointF):
            self._rs = rs
        else:
            self._rs = QPointF(rs[0], rs[1])
        self.updateGeometry()
        self.compute(all=True)

    @property
    def phis(self):
        return self._phis

    @phis.setter
    def phis(self, phis):
        self._phis = phis
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

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = float(beta)
        self.updateGeometry()
        self.compute(all=True)

    @property
    def k0(self):
        return self._k0

    @k0.setter
    def k0(self, k0):
        self._k0 = float(k0)
        self.compute(all=True)
