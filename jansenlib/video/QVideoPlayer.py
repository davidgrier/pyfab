# -*- coding: utf-8 -*-

"""QVideoPlayer.py: pyqtgraph module for OpenCV video playback."""

import cv2
from pyqtgraph.Qt import QtCore
import numpy as np


class QVideoPlayer(QtCore.QObject):
    """OpenCV video player

    Continuously reads frames from a video file,
    emitting sigNewFrame when each frame becomes available.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self,
                 filename=None):
        super(QVideoPlayer, self).__init__()

        if cv2.__version__.startswith('2.'):
            self._WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
            self._LENGTH = cv2.cv.CV_CAP_PROP_FRAME_COUNT
            self._FPS = cv2.cv.CV_CAP_PROP_FPS
        else:
            self._WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
            self._LENGTH = cv2.CAP_PROP_FRAME_COUNT
            self._FPS = cv2.CAP_PROP_FPS

        self.running = False
        self.emitting = False

        if filename is not None:
            self.open(filename)

    def run(self):
        while self.running:
            ready, self.frame = self.capture.read()
            if not ready:
                break
            if self.emitting:
                self.sigNewFrame.emit(self.frame)
                print('emitting')
        self.capture.release()

    @QtCore.pyqtSlot(str)
    def open(self, filename):
        self.capture = cv2.VideoCapture(filename)

    @QtCore.pyqtSlot()
    def close(self):
        self.stop()

    @QtCore.pyqtSlot()
    def start(self):
        if not self.running:
            self.running = True
            self.emitting = True
            self.run()

    @QtCore.pyqtSlot()
    def stop(self):
        self.emitting = False
        self.running = False

    @QtCore.pyqtSlot(bool)
    def pause(self, paused):
        self.emitting = not paused

    @property
    def width(self):
        return int(self.capture.get(self._WIDTH))

    @property
    def height(self):
        return int(self.capture.get(self._HEIGHT))

    @property
    def size(self):
        return QtCore.QSize(self.width, self.height)

    @property
    def length(self):
        return int(self.capture.get(self._LENGTH))

    @property
    def fps(self):
        return int(self.capture.get(self._FPS))

    @property
    def roi(self):
        return QtCore.QRectF(0., 0., self.width, self.height)


if __name__ == '__main__':
    fn = '/Users/grier/data/fabdvr.avi'
    a = QVideoPlayer(fn)
    print(a.length(), a.fps())
    a.capture.release()
