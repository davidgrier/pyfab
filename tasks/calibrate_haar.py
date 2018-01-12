from maxtask import maxtask
from PyQt4.QtGui import QVector3D
import numpy as np


class wavelet_response(maxtask):

    def __init__(self, trap, nframes=10, **kwargs):
        super(wavelet_response, self).__init__(nframes=nframes, **kwargs)
        psi = trap.psi
        psi[0:psi.shape[0]/2, :] *= np.exp(1j * np.pi)


class calibrate_haar(maxtask):

    def __init__(self, nframes=10, **kwargs):
        super(calibrate_haar, self).__init__(nframes=nframes, **kwargs)

    def setParent(self, parent):
        self.parent = parent
        self.parent.pattern.clearTraps()
        xc = 100
        yc = 100
        dim = 10
        self.pos = [QVector3D(xc, yc, 0)]
        self.trap = self.parent.pattern.createTraps(self.pos)
        self.roi = np.ogrid[yc-dim:yc+dim+1, xc-dim:xc+dim+1]

    def dotask(self):
        roi = self.frame[self.roi]
        val = np.sum(roi)
        print(roi.shape, val)
