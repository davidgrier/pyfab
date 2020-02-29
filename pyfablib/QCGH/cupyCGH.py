# -*- coding: utf-8 -*-

"""CUDA-accelerated CGH computation pipeline implemented with cupy"""

from .CGH import CGH
import cupy as cp
import math

cp.cuda.Device()


class cupyCGH(CGH):
    def __init__(self, *args, **kwargs):
        super(cupyCGH, self).__init__(*args, **kwargs)

        self.block = (16, 16, 1)
        self.grid = (math.ceil(self.height / self.block[0]),
                     math.ceil(self.width / self.block[1]))

        self._outeratan2f = cp.RawKernel(r'''
        extern "C" __global__
        void outeratan2f(const double *a, \
                         const double *b, \
                         double *out, \
                         int na, int nb)
        {
          double bj;
          for(int j = threadIdx.y + blockDim.y * blockIdx.y; \
              j < nb; j += blockDim.y * gridDim.y) {
            bj = b[j];
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; \
                i < na; i += blockDim.x * gridDim.x) {
              out[i*nb + j] = atan2f(bj, a[i]);
            }
          }
        }
        ''', 'outeratan2f')

        self._outerhypot = cp.RawKernel(r'''
        extern "C" __global__
        void outerhypot(const double *a, \
                        const double *b, \
                        double *out, \
                        int na, int nb)
        {
          double bj;
          for(int j = threadIdx.y + blockDim.y * blockIdx.y; \
              j < nb; j += blockDim.y * gridDim.y) {
            bj = b[j];
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; \
                i < na; i += blockDim.x * gridDim.x) {
              out[i*nb + j] = hypot(a[i], bj);
            }
          }
        }
        ''', 'outerhypot')

        self._phase = cp.RawKernel(r'''
        # include <cuComplex.h>

        extern "C" __global__
        void phase(cuFloatComplex *psi, \
                   unsigned char *out, \
                   int nx, int ny)
        {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          int j = threadIdx.y + blockDim.y * blockIdx.y;

          int n;
          float phi;
          float argpsi;
          const float RAD2BYTE = 40.743664;

          float real;
          float imag;

          if (i < nx && j < ny){
            n = i*ny + j;
            real = cuCrealf(psi[n]);
            imag = cuCimagf(psi[n]);
            argpsi = atan2(imag, real);
            phi = RAD2BYTE * argpsi + 127.;
            out[n] = (unsigned char) phi;
          }
        }
        ''', 'phase')

    def outeratan2f(self, a, b, out):
        self._outeratan2f(self.grid, self.block,
                          (a, b, out, cp.int32(a.size), cp.int32(b.size)))

    def outerhypot(self, a, b, out):
        self._outerhypot(self.grid, self.block,
                         (a, b, out, cp.int32(a.size), cp.int32(b.size)))

    def phase(self, a, out):
        self._phase(self.grid, self.block,
                    (a, out, cp.int32(a.shape[0]), cp.int32(a.shape[1])))

    def quantize(self, psi):
        #phi = ((128. / cp.pi) * cp.angle(psi) + 127.).astype(cp.uint8)
        # phi.get(out=self.phi)
        self.phase(psi, self._phi)
        self._phi.get(out=self.phi)
        return self.phi

    def compute_displace(self, amp, r, buffer):
        ex = cp.exp(self._iqx * r.x() + self._iqxz * r.z(), dtype=cp.complex64)
        ey = cp.exp(self._iqy * r.y() + self._iqyz * r.z(), dtype=cp.complex64)
        cp.outer(amp * ey, ex, buffer)

    def updateGeometry(self):
        # GPU variables
        self._psi = cp.zeros(self.shape, dtype=cp.complex64)
        self._phi = cp.zeros(self.shape, dtype=cp.uint8)
        self._theta = cp.zeros(self.shape, dtype=cp.float64)
        self._rho = cp.zeros(self.shape, dtype=cp.float64)
        alpha = cp.cos(cp.radians(self.phis, dtype=cp.float64))
        x = alpha*(cp.arange(self.width, dtype=cp.float64) -
                   cp.float64(self.xs))
        y = cp.arange(self.height, dtype=cp.float64) - cp.float64(self.ys)
        qx = self.qprp * x
        qy = self.qprp * y
        self._iqx = (1j * qx).astype(cp.complex64)
        self._iqy = (1j * qy).astype(cp.complex64)
        self._iqxz = (1j * self.qpar * x * x).astype(cp.complex64)
        self._iqyz = (1j * self.qpar * y * y).astype(cp.complex64)
        self.outeratan2f(y, x, self._theta)
        self.outerhypot(qy, qx, self._rho)
        # CPU variables
        self.phi = self._phi.get()
        self.iqx = self._iqx.get()
        self.iqy = self._iqy.get()
        self.theta = self._theta.get()
        self.qr = self._rho.get()
        self.sigUpdateGeometry.emit()

    def bless(self, field):
        if type(field) is complex:
            gpu_field = cp.ones(self.shape, dtype=cp.complex64)
        else:
            gpu_field = cp.asarray(field.astype(cp.complex64))
        return gpu_field
