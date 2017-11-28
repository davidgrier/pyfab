#!/usr/bin/env python

"""QCameraDevice.py: pyqtgraph module for OpenCV video camera."""

import cv2
from pyqtgraph.Qt import QtCore


def is_cv2():
    return cv2.__version__.startswith("2.")


class QCameraThread(QtCore.QThread):
    """Grab frames as fast as possible in a separate thread
    to minimize latency for frame acquisition.
    """

    def __init__(self, camera):
        super(QCameraThread, self).__init__()
        self.camera = camera
        self.keepGrabbing = True

    def __del__(self):
        self.wait()

    def run(self):
        while self.keepGrabbing:
            self.camera.grab()

    def stop(self):
        self.keepGrabbing = False


class QCameraDevice(QtCore.QObject):
    """Low latency OpenCV camera intended to act as an image source
    for PyQt applications.
    """
    _DEFAULT_FPS = 24

    def __init__(self,
                 cameraId=0,
                 size=None,
                 parent=None):
        super(QCameraDevice, self).__init__(parent)

        self.camera = cv2.VideoCapture(cameraId)
        self.thread = QCameraThread(self.camera)

        self.size = size

        try:
            if is_cv2():
                self.fps = int(self.camera.get(cv2.cv.CV_CAP_PROP_FPS))
            else:
                self.fps = int(self.camera.get(cv2.CAP_PROP_FPS))
        except RuntimeError:
            raise
        else:
            self.fps = self._DEFAULT_FPS

    # Reduce latency by continuously grabbing frames in a background thread
    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.thread.stop()

    def close(self):
        self.stop()
        self.camera.release()

    # Read requests return the most recently grabbed frame
    def read(self):
        if self.thread.isRunning():
            ready, frame = self.camera.retrieve()
        else:
            ready, frame = False, None
        return ready, frame

    @property
    def size(self):
        if is_cv2():
            h = int(self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            w = int(self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        else:
            h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        return QtCore.QSize(w, h)

    @size.setter
    def size(self, size):
        if size is None:
            return
        if is_cv2():
            self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, size[1])
            self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, size[0])
        else:
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, fps):
        if (fps > 0):
            self._fps = fps
        else:
            self._fps = self._DEFAULT_FPS

    @property
    def roi(self):
        return QtCore.QRectF(0., 0., self.size.width(), self.size.height())
