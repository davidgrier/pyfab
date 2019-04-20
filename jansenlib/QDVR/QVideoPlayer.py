# -*- coding: utf-8 -*-

"""QVideoPlayer.py: pyqtgraph module for OpenCV video playback."""

import cv2
from PyQt5.QtCore import (QObject, QTimer, QSize, QRectF,
                          pyqtSignal, pyqtSlot, pyqtProperty)
import numpy as np


class QVideoPlayer(QObject):
    """OpenCV video player

    Continuously reads frames from a video file,
    emitting sigNewFrame when each frame becomes available.
    """

    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self,
                 parent=None,
                 filename=None):
        super(QVideoPlayer, self).__init__(parent)

        self.running = False

        if cv2.__version__.startswith('2.'):
            self._SEEK = cv2.cv.CV_CAP_PROP_POS_FRAMES
            self._WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
            self._LENGTH = cv2.cv.CV_CAP_PROP_FRAME_COUNT
            self._FPS = cv2.cv.CV_CAP_PROP_FPS
        else:
            self._SEEK = cv2.CAP_PROP_POS_FRAMES
            self._WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
            self._LENGTH = cv2.CAP_PROP_FRAME_COUNT
            self._FPS = cv2.CAP_PROP_FPS

        self.capture = cv2.VideoCapture(filename)
        if self.capture.isOpened():
            self.delay = 1000. / self.fps
            self.width = int(self.capture.get(self._WIDTH))
            self.height = int(self.capture.get(self._HEIGHT))
        else:
            self.close()

    def isOpened(self):
        return self.capture is not None

    def close(self):
        self.capture.release()
        self.capture = None

    def seek(self, frame):
        self.capture.set(self._SEEK, frame)

    @pyqtSlot()
    def emit(self):
        if not self.running:
            self.close()
            return
        if self.rewinding:
            self.seek(0)
            self.rewinding = False
        if self.emitting:
            ready, self.frame = self.capture.read()
            if ready:
                self.sigNewFrame.emit(self.frame)
            else:
                self.emitting = False
        QTimer.singleShot(self.delay, self.emit)

    @pyqtSlot()
    def start(self):
        if self.running:
            return
        self.running = True
        self.emitting = True
        self.rewinding = False
        self.emit()

    @pyqtSlot()
    def stop(self):
        self.running = False

    @pyqtSlot()
    def rewind(self):
        self.rewinding = True

    @pyqtSlot(bool)
    def pause(self, paused):
        self.emitting = not paused

    def isPaused(self):
        return not self.emitting

    @pyqtProperty(QSize)
    def size(self):
        return QSize(self.width, self.height)

    @pyqtProperty(int)
    def length(self):
        return int(self.capture.get(self._LENGTH))

    @pyqtProperty(int)
    def fps(self):
        return int(self.capture.get(self._FPS))

    @pyqtProperty(QRectF)
    def roi(self):
        return QRectF(0., 0., self.width, self.height)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    fn = '/Users/grier/data/fabdvr.avi'
    a = QVideoPlayer(fn)
    a.start()
    sys.exit(app.exec_())
