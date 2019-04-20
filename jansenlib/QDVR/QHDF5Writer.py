# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot)
import numpy as np
import h5py
import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QHDF5Writer(QObject):

    sigFrameNumber = pyqtSignal(int)
    sigFinished = pyqtSignal()

    def __init__(self, filename, nframes=10000):
        super(QHDF5Writer, self).__init__()

        self.file = h5py.File(filename, 'w')
        self.video = self.file.create_group('images')
        self.start = time.time()
        self.file.attrs.update({'Timestamp': self.start})
        self.framenumber = 0
        self.target = nframes

    @pyqtSlot(np.ndarray)
    def write(self, frame):
        if (self.framenumber >= self.target):
            self.sigFinished.emit()
            return
        now = time.time() - self.start
        self.video.create_dataset(str(now), data=frame)
        self.framenumber += 1
        self.sigFrameNumber.emit(self.framenumber)

    @pyqtSlot()
    def close(self):
        self.file.close()
