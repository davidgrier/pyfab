# -*- coding: utf-8 -*-

"""CUDA-accelerated CGH computation pipeline"""

from PyQt4 import QtCore
from CGH import CGH
import numpy as np

import pycuda.driver as cuda
import pycuda.tools as tools
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.compiler import SourceModule

cuda.init()


class cudaCGH(CGH):

    def __init__(self, **kwargs):
        super(cudaCGH, self).__init__(**kwargs)

    @QtCore.pyqtSlot()
    def start(self):
        self.device = cuda.Device(0)
        self.context = self.device.make_context()

        mod = SourceModule("""
        #include <pycuda-complex.hpp>
        typedef pycuda::complex<float> pyComplex;

        __device__ float arctan(float y, float x){
          const float ONEQTR_PI = 0.78539819;
          const float THRQTR_PI = 2.3561945;
          float r, angle;
          float abs_y = fabs(y) + 1e-10;
          if (x < 0.) {
            r = (x + abs_y) / (abs_y - x);
            angle = THRQTR_PI;
          }
          else {
            r = (x - abs_y) / (x + abs_y);
            angle = ONEQTR_PI;
          }
          angle += (0.1963 * r * r - 0.9817) * r;
          if (y < 0.)
            return(-angle);
          else
            return(angle);
        }

        __global__ void outeratan2f(float *x, \
                                    float *y, \
                                    float *out, \
                                    int nx, int ny)
        {
          float yj;
          for(int j = threadIdx.y + blockDim.y * blockIdx.y; \
              j < ny; j += blockDim.y * gridDim.y) {
            yj = y[j];
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; \
                i < nx; i += blockDim.x * gridDim.x) {
              out[i*ny + j] = atan2f(yj, x[i]);
            }
          }
        }

        __global__ void outerhypot(float *x, \
                                   float *y, \
                                   float *out, \
                                   int nx, int ny)
        {
          float yj;
          for(int j = threadIdx.y + blockDim.y * blockIdx.y; \
              j < ny; j += blockDim.y * gridDim.y) {
            yj = y[j];
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; \
                i < nx; i += blockDim.x * gridDim.x) {
              out[i*ny + j] = hypot(x[i], yj);
            }
          }
        }

        __global__ void outer(pyComplex *a, \
                              pyComplex *b, \
                              pyComplex *out, \
                              int nx, int ny)
        {
          pyComplex bj;
          for(int j = threadIdx.y + blockDim.y * blockIdx.y; \
              j < ny; j += blockDim.y * gridDim.y) {
            bj = b[j];
            for(int i = threadIdx.x + blockDim.x * blockIdx.x; \
                i < nx; i += blockDim.x * gridDim.x) {
              out[i*ny + j] = a[i]*bj;
            }
          }
        }

        __global__ void phase(pyComplex *psi, \
                              unsigned char *out, \
                              int nx, int ny)
        {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          int j = threadIdx.y + blockDim.y * blockIdx.y;

          int n;
          float phi;
          const float RAD2BYTE = 40.743664;

          if (i < nx && j < ny){
            n = i*ny + j;
            phi = RAD2BYTE * arg(psi[n]) + 127.;
            out[n] = (unsigned char) phi;
          }
        }
        """)
        self._outer = mod.get_function('outer')
        self._outeratan2f = mod.get_function('outeratan2f')
        self._outerhypot = mod.get_function('outerhypot')
        self._phase = mod.get_function('phase')
        self.npts = np.int32(self.w * self.h)
        self.block = (16, 16, 1)
        dx, mx = divmod(self.w, self.block[0])
        dy, my = divmod(self.h, self.block[1])
        self.grid = ((dx + (mx > 0)) * self.block[0],
                     (dy + (my > 0)) * self.block[1])
        super(cudaCGH, self).start()

    def outer(self, a, b, out):
        self._outer(a, b, out, np.int32(a.size), np.int32(b.size),
                    block=self.block, grid=self.grid)

    def outeratan2f(self, a, b, out):
        self._outeratan2f(a, b, out, np.int32(a.size), np.int32(b.size),
                          block=self.block, grid=self.grid)

    def outerhypot(self, a, b, out):
        self._outerhypot(a, b, out, np.int32(a.size), np.int32(b.size),
                         block=self.block, grid=self.grid)

    def phase(self, a, out):
        self._phase(a, out, np.int32(a.shape[0]), np.int32(a.shape[1]),
                    block=self.block, grid=self.grid)

    @QtCore.pyqtSlot()
    def stop(self):
        super(cudaCGH, self).stop()
        self.context.pop()
        self.context = None
        tools.clear_context_caches()

    def quantize(self, psi):
        self.phase(psi, self._phi)
        self._phi.get(self.phi)
        return self.phi

    def compute_displace(self, amp, r, buffer):
        cumath.exp(self._iqx * r.x() + self._iqxsq * r.z(), out=self._ex)
        cumath.exp(self._iqy * r.y() + self._iqysq * r.z(), out=self._ey)
        self._ey *= amp
        self.outer(self._ey, self._ex, buffer),

    def updateGeometry(self):
        # GPU storage
        self._psi = gpuarray.zeros(self.shape, dtype=np.complex64)
        self._phi = gpuarray.zeros(self.shape, dtype=np.uint8)
        self._theta = gpuarray.zeros(self.shape, dtype=np.float32)
        self._rho = gpuarray.zeros(self.shape, dtype=np.float32)
        self._ex = gpuarray.zeros(self.w, dtype=np.complex64)
        self._ey = gpuarray.zeros(self.h, dtype=np.complex64)
        # Geometry
        qx = gpuarray.arange(self.w, dtype=np.float32).astype(np.complex64)
        qy = gpuarray.arange(self.h, dtype=np.float32).astype(np.complex64)
        qx = self._qpp * (qx - self.xs)
        qy = self._alpha * self._qpp * (qy - self.ys)
        self._iqx = 1j * qx
        self._iqy = 1j * qy
        self._iqxsq = 1j * qx * qx
        self._iqysq = 1j * qy * qy
        self.outeratan2f(qy.real, qx.real, self._theta)
        self.outerhypot(qy.real, qx.real, self._rho)
        # CPU versions
        self.phi = np.zeros(self.shape, dtype=np.uint8)
        self.iqx = self._iqx.get()
        self.iqy = self._iqy.get()
        self.theta = self._theta.get()
        self.qr = self._rho.get()
        self.sigUpdateGeometry.emit()

    def bless(self, field):
        if type(field) is complex:
            field = np.ones(self.shape)
        self.context.push()
        gpu_field = gpuarray.to_gpu(field.astype(np.complex64))
        cuda.Context.pop()
        return gpu_field
