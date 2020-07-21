# -*- coding: utf-8 -*-

"""CGH.py: compute phase-only holograms for optical traps."""

import numpy as np
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty,
                          QObject, QPointF)
from PyQt5.QtGui import (QVector3D, QMatrix4x4)
from numba import jit
from time import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    sigHologramReady = pyqtSignal(np.ndarray)
    sigUpdateGeometry = pyqtSignal()

    def __init__(self, parent=None, shape=(512, 512)):
        super(CGH, self).__init__(parent)

        # SLM geometry
        self._shape = shape

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
        self.m = QMatrix4x4()
        # Location of optical axis in camera coordinates
        self._rc = QVector3D(320., 240., 0.)
        # Orientation of camera relative to SLM [degrees]
        self._thetac = 0.

        # Location of optical axis in SLM coordinates
        self._rs = QPointF(self.width/2., self.height/2.)
        # Tilt of SLM relative to optical axis [degrees]
        self._phis = 8.

        # Splay wavenumber
        self._splayFactor = 0.01

    # Slots for threaded operation
    @pyqtSlot()
    def start(self):
        logger.info('starting CGH pipeline')
        self.updateGeometry()
        self.updateTransformationMatrix()
        return self

    @pyqtSlot()
    def stop(self):
        logger.info('stopping CGH pipeline')

    @pyqtSlot(object, object)
    def setProperty(self, name, value):
        setattr(self, name, value)

    @pyqtSlot(object)
    def psi(self):
        return self._psi

    # Methods for computing holograms
    @staticmethod
    @jit(nopython=True)
    def quantize(psi):
        """Compute the phase of the field, scaled to uint8"""
        return ((128. / np.pi) * np.angle(psi) + 127.).astype(np.uint8)

    def window(self, r):
        """Adjust amplitude to account for aperture size"""
        x = 0.5 * np.pi * np.array([r.x() / self.width,
                                    r.y() / self.height])
        fac = 1. / np.prod(np.sinc(x))
        return np.min((np.abs(fac), 100.))

    def map_coordinates(self, r):
        """map coordinates into trap space"""
        r = self.m * r
        # axial splay
        fac = 1. / (1. + self.splayFactor * (r.z() - self.rc.z()))
        r *= QVector3D(fac, fac, 1.)
        return r

    # @jit(nopython=True)
    def compute_displace(self, amp, r, buffer):
        """Compute phase hologram to displace a trap with
        a specified complex amplitude to a specified position
        """
        r = self.map_coordinates(r)
        ex = np.exp(self.iqx * r.x() + self.iqxz * r.z())
        ey = np.exp(self.iqy * r.y() + self.iqyz * r.z())
        np.outer(amp * ey, ex, buffer)

    # @jit
    @pyqtSlot(object)
    def compute(self, traps):
        """Compute phase hologram for specified traps"""
        start = time()
        self._psi.fill(0j)
        for trap in traps:
            self._psi += trap.psi
        self.phi = self.quantize(self._psi)
        self.sigHologramReady.emit(self.phi)
        self.time = time() - start

    def bless(self, field):
        """Ensure that field has correct type for compute"""
        if field is None:
            return None
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
        alpha = np.cos(np.radians(self.phis))
        x = alpha*(np.arange(self.width) - self.xs)
        y = np.arange(self.height) - self.ys
        self.iqx = 1j * self.qprp * x
        self.iqy = 1j * self.qprp * y
        self.iqxz = 1j * self.qpar * x * x
        self.iqyz = 1j * self.qpar * y * y
        self.theta = np.arctan2.outer(y, x)
        self.qr = np.hypot.outer(self.qprp * y,
                                 self.qprp * x)
        self.sigUpdateGeometry.emit()

    def updateTransformationMatrix(self):
        """Translate and rotate requested trap positions to account
        for position and orientation of camera relative to SLM
        """
        self.m.setToIdentity()
        self.m.rotate(self.thetac, 0., 0., 1.)
        self.m.translate(-self.rc)
        self.sigUpdateGeometry.emit()

    # Hologram geometry
    @pyqtProperty(int)
    def height(self):
        return self.shape[0]

    @pyqtProperty(int)
    def width(self):
        return self.shape[1]

    @pyqtProperty(list)
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        self.updateGeometry()

    # Derived constants
    @pyqtProperty(float)
    def wavenumber(self):
        '''Wavenumber of trapping light in the medium [radians/um]'''
        return 2.*np.pi*self.refractiveIndex/self.wavelength

    @pyqtProperty(float)
    def qprp(self):
        '''In-plane displacement factor [radians/(pixel phixel)]'''
        cfactor = self.cameraPitch/self.magnification  # [um/pixel]
        sfactor = self.slmPitch/self.scaleFactor       # [um/phixel]
        return (self.wavenumber/self.focalLength)*cfactor*sfactor

    @pyqtProperty(float)
    def qpar(self):
        '''Axial displacement factor [radians/(pixel phixel^2)]'''
        cfactor = self.cameraPitch/self.magnification  # [um/pixel]
        sfactor = self.slmPitch/self.scaleFactor       # [um/phixel]
        return (self.wavenumber/(2.*self.focalLength**2))*cfactor*sfactor**2

    # Calibration constants
    # 1. Instrument parameters
    @pyqtProperty(float)
    def wavelength(self):
        '''Vacuum wavelength of trapping laser [um]'''
        return self._wavelength

    @wavelength.setter
    def wavelength(self, wavelength):
        self._wavelength = wavelength
        self.updateGeometry()

    @pyqtProperty(float)
    def refractiveIndex(self):
        '''Refractive index of medium'''
        return self._refractiveIndex

    @refractiveIndex.setter
    def refractiveIndex(self, refractiveIndex):
        self._refractiveIndex = refractiveIndex
        self.updateGeometry

    @pyqtProperty(float)
    def magnification(self):
        '''Magnification of objective lens'''
        return self._magnification

    @magnification.setter
    def magnification(self, magnification):
        self._magnification = magnification
        self.updateGeometry()

    @pyqtProperty(float)
    def focalLength(self):
        '''Focal length of objective lens [um]'''
        return self._focalLength

    @focalLength.setter
    def focalLength(self, focalLength):
        self._focalLength = focalLength
        self.updateGeometry()

    @pyqtProperty(float)
    def cameraPitch(self):
        '''Pixel pitch of camera [um/pixel]'''
        return self._cameraPitch

    @cameraPitch.setter
    def cameraPitch(self, cameraPitch):
        self._cameraPitch = cameraPitch
        self.updateGeometry()

    @pyqtProperty(float)
    def slmPitch(self):
        '''Pixel pitch of SLM [um/pixel]'''
        return self._slmPitch

    @slmPitch.setter
    def slmPitch(self, slmPitch):
        self._slmPitch = slmPitch
        self.updateGeometry()

    @pyqtProperty(float)
    def scaleFactor(self):
        '''SLM scale factor'''
        return self._scaleFactor

    @scaleFactor.setter
    def scaleFactor(self, scaleFactor):
        self._scaleFactor = scaleFactor
        self.updateGeometry()

    # Calibration constants
    # 2. Camera plane
    @pyqtProperty(float)
    def xc(self):
        '''x coordinate of optical axis in camera plane [pixels]'''
        return self.rc.x()

    @xc.setter
    def xc(self, xc):
        rc = self.rc
        rc.setX(xc)
        self.rc = rc

    @pyqtProperty(float)
    def yc(self):
        '''y coordinate of optical axis in camera plane [pixels]'''
        return self.rc.y()

    @yc.setter
    def yc(self, yc):
        rc = self.rc
        rc.setY(yc)
        self.rc = rc

    @pyqtProperty(float)
    def zc(self):
        '''Axial displacement of trapping plane
        from camera plane [pixels]'''
        return self.rc.z()

    @zc.setter
    def zc(self, zc):
        rc = self.rc
        rc.setZ(zc)
        self.rc = rc

    @pyqtProperty(QVector3D)
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

    @pyqtProperty(float)
    def thetac(self):
        '''Orientation of camera relative to SLM [degrees]'''
        return self._thetac

    @thetac.setter
    def thetac(self, thetac):
        self._thetac = float(thetac)
        self.updateTransformationMatrix()

    # Calibration constants
    # 3. SLM plane
    @pyqtProperty(float)
    def xs(self):
        '''x coordinate of optical axis in SLM plane [pixels]'''
        return self.rs.x()

    @xs.setter
    def xs(self, xs):
        rs = self.rs
        rs.setX(xs)
        self.rs = rs

    @pyqtProperty(float)
    def ys(self):
        '''y coordinate of optical axis in SLM plane [pixels]'''
        return self.rs.y()

    @ys.setter
    def ys(self, ys):
        rs = self.rs
        rs.setY(ys)
        self.rs = rs

    @pyqtProperty(QPointF)
    def rs(self):
        '''Location of optical axis in SLM plane [pixels]'''
        return self._rs

    @rs.setter
    def rs(self, rs):
        if isinstance(rs, QPointF):
            self._rs = rs
        else:
            self._rs = QPointF(rs[0], rs[1])
        self.updateGeometry()

    @pyqtProperty(float)
    def phis(self):
        '''Tilt of SLM relative to optical axis [degrees]'''
        return self._phis

    @phis.setter
    def phis(self, phis):
        self._phis = phis
        self.updateGeometry()

    @pyqtProperty(float)
    def splayFactor(self):
        return self._splayFactor

    @splayFactor.setter
    def splayFactor(self, splayFactor):
        self._splayFactor = float(splayFactor)
        self.updateGeometry()
