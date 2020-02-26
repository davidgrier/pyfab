# -*- coding: utf-8 -*-

"""
QCuCustomTrap.py: Drawing a trap along a parametric curve
                  using CUDA acceleration.
"""

from .QCustomTrap import QCustomTrap
import math
import numpy as np
import cupy as cp


class QCuCustomTrap(QCustomTrap):

    def __init__(self, **kwargs):
        super(QCuCustomTrap, self).__init__(**kwargs)
        self.block = None
        self.grid = None

        self._integrate = cp.RawKernel(r"""
        #include <cuComplex.h>

        extern "C" __global__
        void integrate(const cuFloatComplex *integrand, \
                       cuFloatComplex *out, \
                       float dt, \
                       int nx, int ny, int nt)
        {
            float coeff;
            float sumReal, sumImag, tempReal, tempImag;
            cuFloatComplex intgrnd;
            for (int i = threadIdx.x + blockDim.x * blockIdx.x; \
                 i < nx; i += blockDim.x * gridDim.x) {
                for (int j = threadIdx.y + blockDim.y * blockIdx.y; \
                     j < ny; j += blockDim.y * gridDim.y) {
                    sumReal = 0;
                    sumImag = 0;
                    for (int k = threadIdx.z + blockDim.z * blockIdx.z; \
                         k < nt; k += blockDim.z * gridDim.z) {
                        intgrnd = integrand[k*ny*nx + i*ny + j];
                        if (k == 0 || k == nt-1) {
                            coeff = 1;
                        }
                        else if (k % 2 == 0) {
                            coeff = 2;
                        }
                        else {
                            coeff = 4;
                        }
                        tempReal = cuCrealf(intgrnd) * coeff;
                        tempImag = cuCimagf(intgrnd) * coeff;
                        sumReal += tempReal;
                        sumImag += tempImag;
                    }
                    sumReal *= dt/3;
                    sumImag *= dt/3;
                    out[i*ny + j] = make_cuFloatComplex(sumReal, sumImag);
                }
            }
        }
        """, "integrate")

    def getBuffers(self, t):
        self.block = (16, 16, 1)
        self.grid = (math.ceil(self.cgh.height / self.block[0]),
                     math.ceil(self.cgh.width / self.block[1]))
        structure = cp.zeros(self.cgh.shape, np.complex64)
        integrand = cp.ones((t.size,
                             self.cgh.shape[0],
                             self.cgh.shape[1]),
                            dtype=np.complex64)
        alpha = np.cos(np.radians(self.cgh.phis))
        x = alpha*(np.arange(self.cgh.width) - self.cgh.xs)
        y = np.arange(self.cgh.height) - self.cgh.ys
        x, y = (cp.asarray(x, dtype=cp.float32),
                cp.asarray(y, dtype=np.float32))
        xv, yv = cp.meshgrid(x, y)
        return structure, integrand, xv, yv

    @staticmethod
    def integrand(t, x, y, S_T, L, rho, m, f, lamb,
                  x_0, y_0, z_0, S, dx_0, dy_0, buff):
        buff *= cp.exp(1.j * (y * x_0 - x * y_0) / rho**2
                       + 1.j * 2*np.pi * m * S / S_T)
        buff *= cp.exp(1.j*np.pi * z_0 *
                       ((x - x_0)**2 + (y - y_0)**2)
                       / (lamb * f**2))
        buff *= cp.sqrt(dx_0**2 + dy_0**2) / L

    def integrate(self, integrand, structure, t, shape):
        nx, ny = shape
        nt = t.size
        dt = cp.float32((t[-1] - t[0]) / t.size)
        self._integrate(self.grid, self.block,
                        (integrand, structure, dt,
                         cp.int32(nx), cp.int32(ny), cp.int32(nt)))
        structure = cp.asnumpy(structure)
        integrand = None
