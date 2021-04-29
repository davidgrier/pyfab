# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot)
import numpy as np
import cv2

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QVideoWriter(QObject):

    sigFrameNumber = pyqtSignal(int)
    sigFinished = pyqtSignal()

    def __init__(self, filename, shape,
                 nframes=10000,
                 fps=24,
                 codec=None):
        super(QVideoWriter, self).__init__()

        self.shape = shape
        color = (len(self.shape) == 3)
        h, w = self.shape[0:2]

        if codec is None:
            # NOTE: libavcodec appears to seg fault when
            # recording with the lossless FFV1 codec
            # codec = 'FFV1'

            # NOTE: HuffyYUV appears to work on both
            # Ubuntu and Macports
            codec = 'HFYU'

        if cv2.__version__.startswith('2.'):
            fourcc = cv2.cv.CV_FOURCC(*codec)
            self.conversion = cv2.cv.CV_COLOR_BGR2RGB
        else:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.conversion = cv2.COLOR_BGR2RGB

        msg = 'Recording: {}x{}, color: {}, fps: {}'
        logger.info(msg.format(w, h, color, fps))
        self.writer = cv2.VideoWriter(filename,
                                      fourcc, fps, (w, h), color)
        self.framenumber = 0
        self.target = nframes
        self.sigFrameNumber.emit(self.framenumber)

    @pyqtSlot(np.ndarray)
    def write(self, frame):
        if ((frame.shape != self.shape) or
                (self.framenumber >= self.target)):
            self.sigFinished.emit()
            return
        if (frame.ndim == 3):
            frame = cv2.cvtColor(frame, self.conversion)
        self.writer.write(frame)
        self.framenumber += 1
        self.sigFrameNumber.emit(self.framenumber)

    @pyqtSlot()
    def close(self):
        self.writer.release()
