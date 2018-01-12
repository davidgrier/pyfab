from task import task
from maxtask import maxtask
from PyQt4.QtGui import QVector3D
import numpy as np
import matplotlib.pyplot as plt


class haar_cleanup(task):

    def __init__(self, **kwargs):
        super(haar_cleanup, self).__init__(**kwargs)
        self.data = []

    def initialize(self):
        self.parent.pattern.clearTraps()
        self.done = True

    def addData(self, data):
        print(data)
        self.data.append(data)

    def dotask(self):
        data = np.array(self.data)
        plt.scatter(data[:,0], data[:,1])
        plt.show()
        print('done')


class wavelet_response(maxtask):

    def __init__(self, trap, cgh, slm, roi0, roi1, roi2,
                 summary, val=128., **kwargs):
        super(wavelet_response, self).__init__(**kwargs)
        self.trap = trap
        self.cgh = cgh
        self.slm = slm
        self.roi0 = roi0
        self.roi1 = roi1
        self.roi2 = roi2
        self.summary = summary
        self.val = val

    def initialize(self):
        psi = self.trap.psi
        psi[0:psi.shape[0]/2,:] *= np.exp(1j * np.pi * self.val / 128.)
        self.slm.data = self.cgh.quantize(psi)

    def dotask(self):
        v1 = np.sum(self.frame[self.roi1]).astype(float)
        v0 = np.sum(self.frame[self.roi0]).astype(float)
        bg = np.sum(self.frame[self.roi2]).astype(float)
        self.summary.addData([self.val, (v1-bg)/(v0-bg)])


class calibrate_haar(task):

    def __init__(self, **kwargs):
        super(calibrate_haar, self).__init__(**kwargs)

    def initialize(self):
        self.parent.pattern.clearTraps()
        cgh = self.parent.cgh
        slm = self.parent.slm
        dim = 15
        xc = np.round(self.parent.cgh.rc.x()).astype(int)
        yc = np.round(self.parent.cgh.rc.y()).astype(int)
        roi0 = np.ogrid[yc-dim:yc+dim+1, xc-dim:xc+dim+1]
        xc = 100
        yc = 100
        pos = [QVector3D(xc, yc, 0)]
        self.parent.pattern.createTraps(pos)
        trap = self.parent.pattern.flatten()[0]
        roi1 = np.ogrid[yc-dim:yc+dim+1, xc-dim:xc+dim+1]
        xc += 100
        roi2 = np.ogrid[yc-dim:yc+dim+1, xc-dim:xc+dim+1]
        summary = haar_cleanup()
        for val in range(0, 255, 2):
            task = wavelet_response(trap, cgh, slm,
                                    roi0, roi1, roi2,
                                    summary, val=val,
                                    delay=5, nframes=10)
            self.parent.tasks.registerTask(task)
        self.parent.tasks.registerTask(summary)
        self.done = True
