# -*- coding: utf-8 -*-

"""CUDA-accelerated CGH computation pipeline implemented with cupy"""

from .CGH import CGH
import cupy as cp


class cupyCGH(CGH):
    def __init__(self, *args, **kwargs):
        super(cudaCGH, self).__init__(*args, **kwargs)

    def quantize(self, psi):
        pass

    def compute_displace(self, amp, r, buffer):
        pass

    def updateGeometry(self):
        self._psi = cp.zeros(self.shape, dtype=cp.complex_)
        alpha = cp.cos(cp.radians(self.phis))
        x = alpha*(cp.arange(self.width) - self.xs)
        y = cp.arange(self.height) - self.ys
        self.iqx = 1j * self.qprp * x
        self.iqy = 1j * self.qprp * y
        self.iqxz = 1j * self.qpar * x * x
        self.iqyz = 1j * self.qpar * y * y
        self.theta = cp.arctan2.outer(y, x)
        self.qr = cp.hypot.outer(self.qprp * y,
                                 self.qprp * x)
        self.sigUpdateGeometry.emit()
        pass

    def bless(self, field):
        pass
