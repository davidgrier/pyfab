# -*- coding: utf-8 -*-

"""QOpenCVThread: OpenCV video camera running in a QThread"""

from OpenCVCamera import OpenCVCamera
from PyQt5.QtCore import (QThread, pyqtSignal, pyqtProperty)
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QOpenCVThread(QThread):
    """OpenCV camera

    Continuously captures frames from a video camera,
    emitting sigNewFrame when each frame becomes available.

    NOTE: Subclassing QThread is appealing for this application
    because reading frames is blocking and I/O-bound, but not
    computationally expensive.  QThread moves the read operation
    into a separate thread via the overridden run() method
    while other methods and properties remain available in
    the calling thread.  This simplifies getting and setting
    the camera's properties.
    """

    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, **kwargs):
        super(QOpenCVThread, self).__init__(parent)
        self.device = OpenCVCamera(**kwargs)
        self.read = self.device.read

    def __del__(self):
        del self.device

    def run(self):
        self.running = True
        while self.running:
            ready, frame = self.read()
            if ready:
                self.sigNewFrame.emit(frame)
            else:
                logger.warn('Failed to read frame')

    def stop(self):
        self.running = False

    @pyqtProperty(bool)
    def flipped(self):
        return self.device.flipped

    @flipped.setter
    def flipped(self, state):
        self.device.flipped = state

    @pyqtProperty(bool)
    def mirrored(self):
        return self.device.mirrored

    @mirrored.setter
    def mirrored(self, state):
        self.device.mirrored = state

    @pyqtProperty(bool)
    def gray(self):
        return self.device.gray

    @gray.setter
    def gray(self, state):
        self.device.gray = state

    @pyqtProperty(int)
    def width(self):
        return self.device.width

    @width.setter
    def width(self, value):
        self.device.width = value

    @pyqtProperty(int)
    def height(self):
        return self.device.height

    @height.setter
    def height(self, value):
        self.device.height = value
