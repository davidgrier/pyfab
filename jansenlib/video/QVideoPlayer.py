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

        self.paused = False
        self.open(filename)

    def emit(self):
        if self.paused:
            return
        ready, self.frame = self.capture.read()
        if ready:
            self.sigNewFrame.emit(self.frame)
            self.framenumber += 1
            print(self.framenumber)

    def open(self, filename):
        self.filename = filename
        self.capture = cv2.VideoCapture(self.filename)

    @QtCore.pyqtSlot()
    def start(self):
        if self.capture is None:
            return
        self._timer = QtCore.QTimer()
        self._timer.timeout.connect(self.emit)
        self._timer.start(1000. / self.fps)
        self.paused = False
        self.framenumber = 0

    @QtCore.pyqtSlot()
    def stop(self):
        self._timer.stop()
        self.capture.release()

    @QtCore.pyqtSlot(bool)
    def pause(self, paused):
        self.paused = paused

    @QtCore.pyqtSlot()
    def rewind(self):
        self.capture.release()
        self.open(self.filename)

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
