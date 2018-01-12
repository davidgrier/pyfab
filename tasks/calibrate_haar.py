from task import task
from maxtask import maxtask
from PyQt4.QtGui import QVector3D
import numpy as np


class haar_cleanup(task):

    def __init__(self, **kwargs):
        super(haar_cleanup, self).__init__(**kwargs)
        self.data = []

    def initialize(self):
        self.parent.pattern.clearTraps()
        self.done = True

    def addData(self, data):
        self.data.append(data)

    def dotask(self):
        print(self.data)


class wavelet_response(maxtask):

    def __init__(self, trap, cgh, slm, roi, summary,
                 val=128., **kwargs):
        super(wavelet_response, self).__init__(**kwargs)
        self.trap = trap
        self.cgh = cgh
        self.slm = slm
        self.roi = roi
        self.summary = summary
        self.val = val

    def initialize(self):
        psi = self.trap.psi
        psi[0:psi.shape[0]/2, :] *= np.exp(1j * np.pi * self.val / 256.)
        self.slm.data = self.cgh.quantize(psi)

    def dotask(self):
        roi = self.frame[self.roi]
        inten = np.sum(roi)
        self.summary.addData([self.val, inten])
        print(self.val, inten)


class calibrate_haar(task):

    def __init__(self, **kwargs):
        super(calibrate_haar, self).__init__(**kwargs)

    def initialize(self):
        self.parent.pattern.clearTraps()
        xc = 100
        yc = 100
        dim = 10
        pos = [QVector3D(xc, yc, 0)]
        self.parent.pattern.createTraps(pos)
        trap = self.parent.pattern.flatten()[0]
        cgh = self.parent.cgh
        slm = self.parent.slm
        roi = np.ogrid[yc-dim:yc+dim+1, xc-dim:xc+dim+1]
        summary = haar_cleanup()
        for val in range(0, 255, 10):
            task = wavelet_response(trap, cgh, slm, roi,
                                    summary,
                                    val=val, nframes=5)
            self.parent.tasks.registerTask(task)
        self.parent.tasks.registerTask(summary)
        self.done = True
