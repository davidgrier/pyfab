from CGH import CGH
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np


class cudaCGH(CGH):

    def __init__(self, slm=None):
        super(cudaCGH, self).__init__(slm=slm)

    def compute_one(self):
        self._psi *= 0. + 0j
        

    def updateGeometry(self):
        print('updating')
        shape = (self.w, self.h)
        self._buffer = gpuarray.zeros(shape, dtype=np.complex64)
        self._psi = gpuarray.zeros(shape, dtype=np.complex64)
        qx = gpuarray.arange(self.w, dtype=np.float64) - self.rs.x()
        qy = gpuarray.arange(self.h, dtype=np.float64) - self.rs.y()
        qx *= self.qpp
        qy *= self.qpp * qy
        self.iqx = 1j * qx
        self.iqy = 1j * qy
        self.iqxsq = 1j * qx * qx
        self.iqysq = 1j * qy * qy
        print('done')

        
if __name__ == '__main__':
    from PyQt4.QtGui import QApplication
    import sys
    from QSLM import QSLM

    app = QApplication(sys.argv)
    slm = QSLM()
    cgh = cudaCGH(slm)
    sys.exit(app.exec_())
