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
        # include <pycuda-complex.hpp>
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

        __global__ void outertheta(float *x, \
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

        __global__ void outerrho(float *x, \
                                 float *y, \
                                 float *out, \
                                 int nx, int ny)
        {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          int j = threadIdx.y + blockDim.y * blockIdx.y;
          if (i < nx && j < ny) {
            out[i*ny + j] = hypotf(x[i], y[j]);
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
        self.outer = mod.get_function('outer')
        self.phase = mod.get_function('phase')
        self.outertheta = mod.get_function('outertheta')
        self.outerrho = mod.get_function('outerrho')
        self.npts = np.int32(self.w * self.h)
        self.block = (16, 16, 1)
        dx, mx = divmod(self.w, self.block[0])
        dy, my = divmod(self.h, self.block[1])
        self.grid = ((dx + (mx > 0)) * self.block[0],
                     (dy + (my > 0)) * self.block[1])
        super(cudaCGH, self).start()

    @QtCore.pyqtSlot()
    def stop(self):
        super(cudaCGH, self).stop()
        self.context.pop()
        self.context = None
        tools.clear_context_caches()

    def quantize(self, psi):
        self.phase(psi, self._phi,
                   np.int32(self.w), np.int32(self.h),
                   block=self.block, grid=self.grid)
        self._phi.get(self.phi)
        return self.phi.T

    def compute_displace(self, amp, r, buffer):
        cumath.exp(self._iqx * r.x() + self._iqxsq * r.z(), out=self._ex)
        cumath.exp(self._iqy * r.y() + self._iqysq * r.z(), out=self._ey)
        self._ex *= amp
        self.outer(self._ex, self._ey, buffer,
                   np.int32(self.w), np.int32(self.h),
                   block=self.block, grid=self.grid)

    def updateGeometry(self):
        shape = (self.w, self.h)
        self._psi = gpuarray.zeros(shape, dtype=np.complex64)
        self._phi = gpuarray.zeros(shape, dtype=np.uint8)
        self._theta = gpuarray.zeros(shape, dtype=np.float32)
        self._rho = gpuarray.zeros(shape, dtype=np.float32)
        self._ex = gpuarray.zeros(self.w, dtype=np.complex64)
        self._ey = gpuarray.zeros(self.h, dtype=np.complex64)
        qx = gpuarray.arange(self.w, dtype=np.float32).astype(np.complex64)
        qy = gpuarray.arange(self.h, dtype=np.float32).astype(np.complex64)
        qx = self._qpp * (qx - self.xs)
        qy = self._alpha * self._qpp * (qy - self.ys)
        self.phi = np.zeros(shape, dtype=np.uint8)
        self._iqx = 1j * qx
        self._iqy = 1j * qy
        self._iqxsq = 1j * qx * qx
        self._iqysq = 1j * qy * qy
        self.iqx = self._iqx.get()
        self.iqy = self._iqy.get()
        self.outertheta(qx, qy, self._theta,
                        np.int32(self.w), np.int32(self.h),
                        block=self.block, grid=self.grid)
        self.outerrho(qx, qy, self._rho,
                      np.int32(self.w), np.int32(self.h),
                      block=self.block, grid=self.grid)
        self.theta = self._theta.get()
        self.qr = self._rho.get()

    def bless(self, field):
        self.context.push()
        gpu_field = gpuarray.to_gpu(field.astype(np.complex64))
        cuda.Context.pop()
        return gpu_field
