# -*- coding: utf-8 -*-

"""CUDA-accelerated CGH computation pipeline implemented with cupy"""

from .CGH import CGH
import cupy as cp
import math


class cupyCGH(CGH):
    def __init__(self, *args, **kwargs):
        super(cupyCGH, self).__init__(*args, **kwargs)

        self.block = (16, 16, 1)
        self.grid = (math.ceil(self.height / self.block[0]),
                     math.ceil(self.width / self.block[1]))

        self._outeratan2f = cp.RawKernel(r'''
        extern "C" __global__
        void outeratan2f(const float *a, \
                         const float *b, \
                         float *out, \
                         int na, int nb)
        {
          float bj;
          for(int j = threadIdx.b + blockDim.b * blockIdx.b; \
              j < nb; j += blockDim.b * gridDim.b) {
            bj = b[j];
            for(int i = threadIdx.a + blockDim.a * blockIdx.a; \
                i < na; i += blockDim.a * gridDim.a) {
              out[i*nb + j] = atan2f(bj, a[i]);
            }
          }
        }
        ''', 'outeratan2f')

        self._outerhypot = cp.RawKernel(r'''
        extern "C" __global__
        void outerhypot(const float *a, \
                        const float *b, \
                        float *out, \
                        int na, int nb)
        {
          float bj;
          for(int j = threadIdx.b + blockDim.b * blockIdx.b; \
              j < nb; j += blockDim.b * gridDim.b) {
            bj = b[j];
            for(int i = threadIdx.a + blockDim.a * blockIdx.a; \
                i < na; i += blockDim.a * gridDim.a) {
              out[i*nb + j] = hypot(a[i], bj);
            }
          }
        }
        ''', 'outerhypot')

    def outeratan2f(self, a, b, out):
        self._outeratan2f(self.grid, self.block,
                          a, b, out, cp.int32(a.size), cp.int32(b.size))

    def outerhypot(self, a, b, out):
        self._outerhypot(self.grid, self.block,
                         a, b, out, cp.int32(a.size), cp.int32(b.size))

    def quantize(self, psi):
        return ((128. / cp.pi) * cp.angle(psi) + 127.).astype(cp.uint8)

    def compute_displace(self, amp, r, buffer):
        ex = cp.exp(self.iqx * r.x() + self.iqxz * r.z())
        ey = cp.exp(self.iqy * r.y() + self.iqyz * r.z())
        cp.outer(amp * ey, ex, buffer)

    def updateGeometry(self):
        self._psi = cp.zeros(self.shape, dtype=cp.complex_)
        alpha = cp.cos(cp.radians(self.phis))
        x = alpha*(cp.arange(self.width) - self.xs)
        y = cp.arange(self.height) - self.ys
        self.iqx = 1j * self.qprp * x
        self.iqy = 1j * self.qprp * y
        self.iqxz = 1j * self.qpar * x * x
        self.iqyz = 1j * self.qpar * y * y
        self.theta = self.outerarctan2(y, x)
        self.qr = self.outerhypot(self.qprp * y, self.qprp * x)
        self.sigUpdateGeometry.emit()
        pass

    def bless(self, field):
        pass
