from CGH import CGH
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


class cudaCGH(CGH):

    def __init__(self, slm=None):
        super(cudaCGH, self).__init__(slm=slm)
        self.init_cuda()

    def init_cuda(self):
        mod = SourceModule("""
        #include <pycuda-complex.hpp>

        __global__ void outer(pycuda::complex<float> *x, \
                              pycuda::complex<float> *y, \
                              pycuda::complex<float> *out, \
                              int nx, int ny)
        {
          int i = threadIdx.x + blockDim.x * blockIdx.x;
          int j = threadIdx.y + blockDim.y * blockIdx.y;
          if (i < nx && j < ny){
            out[i*ny + j] = x[i]*y[j];
          }
        }

        __global__ void phase(pycuda::complex<float> *psi, \
                              unsigned char *out, \
                              int nx, int ny)
        {
          int i = threadIdx.x + blockIdx.x * blockDim.x;
          int j = threadIdx.y + blockIdx.y * blockDim.y;
          if (i < nx && j < ny){
            int n = i*ny + j;
            float im = psi[n]._M_im;
            float re = psi[n]._M_re;
            float phi = (128./3.14159265359) * atan2f(im, re) + 127.;
            out[n] = (unsigned char) phi;
          }
        }
        """)
        self.outer = mod.get_function("outer")
        self.phase = mod.get_function("phase")
        self.npts = np.int32(self.w * self.h)
        self.block = (16, 16, 1)
        dx, mx = divmod(self.w, self.block[0])
        dy, my = divmod(self.h, self.block[1])
        self.grid = ((dx + (mx > 0)) * self.block[0],
                     (dy + (my > 0)) * self.block[1])

    def quantize(self):
        self.phase(self._psi, self._phi,
                   np.int32(self.w), np.int32(self.h),
                   block=self.block, grid=self.grid)
        self._phi.get(self.phi)
        return self.phi.T
    
    def compute_one(self, amp, x, y, z):
        cumath.exp(self.iqx * x + self.iqxsq * z, out=self._ex)
        cumath.exp(self.iqy * y + self.iqysq * z, out=self._ey)
        self._ex *= amp
        self.outer(self._ex, self._ey, self._buffer,
                   np.int32(self.w), np.int32(self.h),
                   block=self.block, grid=self.grid)
        return self._buffer

    def updateGeometry(self):
        shape = (self.w, self.h)
        self._buffer = gpuarray.zeros(shape, dtype=np.complex64)
        self._psi = gpuarray.zeros(shape, dtype=np.complex64)
        self._phi = gpuarray.zeros(shape, dtype=np.uint8)
        self.phi = np.zeros(shape, dtype=np.uint8)
        self._ex = gpuarray.zeros(self.w, dtype=np.complex64)
        self._ey = gpuarray.zeros(self.h, dtype=np.complex64)
        qx = gpuarray.arange(self.w, dtype=np.float32).astype(np.complex64)
        qy = gpuarray.arange(self.h, dtype=np.float32).astype(np.complex64)
        qx = self.qpp * (qx - self.rs.x())
        qy = self.qpp * (qy - self.rs.y())
        self.iqx = 1j * qx
        self.iqy = 1j * qy
        self.iqxsq = 1j * qx * qx
        self.iqysq = 1j * qy * qy

        
if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    import sys
    from QSLM import QSLM

    app = QApplication(sys.argv)
    slm = QSLM()
    cgh = cudaCGH(slm)
    sys.exit(app.exec_())
