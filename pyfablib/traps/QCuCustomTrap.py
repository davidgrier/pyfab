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

        self._mod = cp.RawModule(code=r"""
        #include <cuComplex.h>

        extern "C" __device__
        cuFloatComplex cuCexpf(cuFloatComplex a) {
            cuFloatComplex phase, out;
            float mod;
            
            mod = expf(cuCrealf(a));
            phase = cuCaddf(make_cuFloatComplex(cosf(cuCimagf(a)), 0.), \
                            make_cuFloatComplex(0., sinf(cuCimagf(a))));
            out = cuCmulf(make_cuFloatComplex(mod, 0.), phase);
            return out;
        }

        extern "C" __device__
        cuFloatComplex integrand(int i, int j, int k, int ny, \
                                 const float *x, const float *y, \
                                 float S_T, float L, float rho, float m, \
                                 float f, float lamb, \
                                 float *x_0, float *y_0, float *z_0, \
                                 float *S, float *dx_0, float *dy_0) {

            const float PI = 3.14159265;

            cuFloatComplex temp, intgrnd;
            temp = make_cuFloatComplex((y[i*ny+j] * x_0[k] - x[i*ny+j] * y_0[k]) / (rho*rho), 0.);
            intgrnd = cuCexpf(cuCmulf(make_cuFloatComplex(0., 1.), temp));
            temp = make_cuFloatComplex(2*PI * m * S[k] / S_T, 0.);
            intgrnd = cuCmulf(intgrnd, cuCexpf(cuCmulf(make_cuFloatComplex(0., 1.), temp)));
            temp = make_cuFloatComplex(PI * z_0[k] * \
                                       ((powf(x[i*ny+j] - x_0[k], 2) + \
                                         powf(y[i*ny+j] - y_0[k], 2)) / (lamb*f*f)), 0.);
            intgrnd = cuCmulf(intgrnd, cuCexpf(cuCmulf(make_cuFloatComplex(0., 1.), temp)));
            temp = make_cuFloatComplex(sqrtf(powf(dx_0[k], 2.) + \
                                             powf(dy_0[k], 2)) / L, 0.);
            intgrnd = cuCmulf(intgrnd, temp);

            return intgrnd;
        }

        extern "C" __global__
        void integrate(const float *x, const float *y, \
                       float S_T, float L, float rho, float m, \
                       float f, float lamb, \
                       float *x_0, float *y_0, float *z_0, \
                       float *S, float *dx_0, float *dy_0,
                       cuFloatComplex *out, \
                       float dt, \
                       int nx, int ny, int nt)
        {
            float coeff, sumReal, sumImag, tempReal, tempImag;
            cuFloatComplex intgrnd;

            for (int i = threadIdx.x + blockDim.x * blockIdx.x; \
                 i < nx; i += blockDim.x * gridDim.x) {
                for (int j = threadIdx.y + blockDim.y * blockIdx.y; \
                     j < ny; j += blockDim.y * gridDim.y) {
                    sumReal = 0;
                    sumImag = 0;
                    for (int k = threadIdx.z + blockDim.z * blockIdx.z; \
                         k < nt; k += blockDim.z * gridDim.z) {
                        intgrnd = integrand(i, j, k, ny, x, y, \
                                            S_T, L, rho, m, f, lamb, \
                                            x_0, y_0, z_0, S, dx_0, dy_0);
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
        """)

        self._integrate = self._mod.get_function('integrate')

    def getBuffers(self, t):
        self.block = (16, 16, 1)
        self.grid = (math.ceil(self.cgh.height / self.block[0]),
                     math.ceil(self.cgh.width / self.block[1]))
        structure = cp.zeros(self.cgh.shape, np.complex64)
        alpha = np.cos(np.radians(self.cgh.phis))
        x = alpha*(np.arange(self.cgh.width) - self.cgh.xs)
        y = np.arange(self.cgh.height) - self.cgh.ys
        x, y = (cp.asarray(x, dtype=cp.float32),
                cp.asarray(y, dtype=np.float32))
        xv, yv = cp.meshgrid(x, y)
        return structure, xv, yv

    def integrate(self, t, x, y, S_T, L, rho, m, f, lamb,
                  x_0, y_0, z_0, S, dx_0, dy_0, out, shape):
        # Get shape and step size
        nx, ny = shape
        nt = t.size
        dt = cp.float32((t[-1] - t[0]) / t.size)
        # Type cast
        S_T, L = (cp.float32(S_T), cp.float32(L))
        rho, m = (cp.float32(rho), cp.float32(m))
        f, lamb = (cp.float32(f), cp.float32(lamb))
        x_0, y_0, z_0 = (cp.asarray(x_0, dtype=cp.float32),
                         cp.asarray(y_0, dtype=cp.float32),
                         cp.asarray(z_0, dtype=cp.float32))
        dx_0, dy_0, S = (cp.asarray(dx_0, dtype=cp.float32),
                         cp.asarray(dy_0, dtype=cp.float32),
                         cp.asarray(S, dtype=cp.float32))
        # Integrate
        self._integrate(self.grid, self.block,
                        (x, y, S_T, L, rho, m, f, lamb,
                         x_0, y_0, z_0, S, dx_0, dy_0,
                         out, dt, nx, ny, nt))
        out = cp.asnumpy(out)
