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
    finished = QtCore.pyqtSignal()

    def __init__(self,
                 filename=None):
        super(QVideoPlayer, self).__init__()

        self.filename = filename

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
        self.emitting = True

    def open(self):
        self.capture = cv2.VideoCapture(self.filename)
        self.delay = 1000. / self.fps
        self.width = int(self.capture.get(self._WIDTH))
        self.height = int(self.capture.get(self._HEIGHT))

    def close(self):
        self.capture.release()

    @QtCore.pyqtSlot()
    def emit(self):
        if not self.running:
            return
        ready, self.frame = self.capture.read()
        if ready:
            self.sigNewFrame.emit(self.frame)
            QtCore.QTimer.singleShot(self.delay, self.emit)
        else:
            self.finished.emit()
            print('eof')

    @QtCore.pyqtSlot()
    def rewind(self):
        self.stop()
        self.start()

    @QtCore.pyqtSlot()
    def start(self):
        if not self.running:
            self.open()
            self.running = True
            self.emit()

    @QtCore.pyqtSlot()
    def stop(self):
        self.running = False
        self.close()

    @QtCore.pyqtSlot(bool)
    def pause(self, paused):
        self.emitting = not paused

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
