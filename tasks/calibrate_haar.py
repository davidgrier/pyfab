from task import task
from maxtask import maxtask
from cleartraps import cleartraps
from createtrap import createtrap
from PyQt4.QtGui import QVector3D
import numpy as np
import matplotlib.pyplot as plt


class haar_summary(task):

    def __init__(self, **kwargs):
        super(haar_summary, self).__init__(**kwargs)
        self.background = 0.
        self.data = []

    def initialize(self):
        print('summary')
        
    def append(self, data):
        print(data)
        self.data.append(data)

    def dotask(self):
        data = np.array(self.data)
        plt.scatter(data[:,0], data[:,1])
        plt.show()
        print('done')

        
class background(maxtask):

    def __init__(self, roi, summary, **kwargs):
        super(background, self).__init__(**kwargs)
        self.roi = roi
        self.summary = summary

    def initialize(self):
        print('background')

    def dotask(self):
        self.summary.background = np.sum(self.frame[self.roi]).astype(float)
        print('background', self.summary.background)

        
class wavelet_response(maxtask):

    def __init__(self, roi, summary, val, **kwargs):
        super(wavelet_response, self).__init__(**kwargs)
        self.roi = roi
        self.summary = summary
        self.val = val

    def initialize(self):
        trap = self.parent.pattern.flatten()[0]
        psi = trap.psi
        psi[0:psi.shape[0]/2,:] *= np.exp(1j * np.pi * self.val / 128.)
        self.parent.slm.data = self.parent.cgh.quantize(psi)

    def dotask(self):
        v = np.sum(self.frame[self.roi]).astype(float)
        bg = self.summary.background
        self.summary.append([self.val, v-bg])


class calibrate_haar(task):

    def __init__(self, **kwargs):
        super(calibrate_haar, self).__init__(**kwargs)

    def initialize(self):
        dim = 15
        xc = 100
        yc = 100
        roi = np.ogrid[yc-dim:yc+dim+1, xc-dim:xc+dim+1]
        summary = haar_summary()
        register = self.parent.tasks.registerTask
        register(cleartraps())
        register(background(roi, summary, delay=5, nframes=60))
        register(createtrap(xc, yc))
        for val in range(0, 255, 5):
            register(wavelet_response(roi, summary, val,
                                      delay=5, nframes=60))
        register(summary)
        register(cleartraps())
