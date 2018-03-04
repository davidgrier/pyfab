# -*- coding: utf-8 -*-

"""QVideoPlayer.py: pyqtgraph module for OpenCV video playback."""

import cv2
from pyqtgraph.Qt import QtCore
import numpy as np
import time


class QVideoPlayer(QtCore.QObject):
    """OpenCV video player

    Continuously reads frames from a video file,
    emitting sigNewFrame when each frame becomes available.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self,
                 filename=None):
        super(QVideoPlayer, self).__init__()

        self.filename = filename
        self.capture = cv2.VideoCapture(self.filename)

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

        self.delay = 1. / self.fps
        self.running = False
        self.emitting = True
        self.rewinding = False

    def run(self):
        while self.running:
            if self.rewinding:
                self.capture.release()
                self.capture = cv2.VideoCapture(self.filename)
                self.rewinding = False
            ready, self.frame = self.capture.read()
            if ready and self.emitting:
                self.sigNewFrame.emit(self.frame)
                self.framenumber += 1
                print(self.framenumber)
            time.sleep(self.delay)
        self.capture.release()

    @QtCore.pyqtSlot()
    def start(self):
        if not self.running:
            self.running = True
            self.framenumber = 0
            self.run()

    @QtCore.pyqtSlot()
    def stop(self):
        self.running = False

    @QtCore.pyqtSlot(bool)
    def pause(self, paused):
        self.emitting = not paused

    @QtCore.pyqtSlot()
    def rewind(self):
        self.rewinding = True

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
    import sys
    from PyQt4 import QtGui

    app = QtGui.QApplication(sys.argv)
    fn = '/Users/grier/data/fabdvr.avi'
    a = QVideoPlayer(fn)
    a.start()
    sys.exit(app.exec_())
