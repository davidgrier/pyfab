#!/usr/bin/env python

"""QCameraItem.py: pyqtgraph module for OpenCV video camera."""

import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from PyQt4.QtCore import Qt
import numpy as np


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

        # if is_cv2():
        #    self.fps = int(self.camera.get(cv2.cv.CV_CAP_PROP_FPS))
        # else:
        #    self.fps = int(self.camera.get(cv2.CAP_PROP_FPS))
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
        return QtCore.QSizeF(w, h)

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


class QVideoItem(pg.ImageItem):
    """Video source for pyqtgraph applications.
    Acts like an ImageItem that periodically polls
    a camera for updated video frames.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, device=None, parent=None,
                 mirrored=True,
                 flipped=True,
                 transposed=False,
                 gray=False,
                 **kwargs):
        pg.setConfigOptions(imageAxisOrder='row-major')
        super(QVideoItem, self).__init__(parent)

        if device is None:
            self.device = QCameraDevice(**kwargs).start()
        else:
            self.device = device.start()

        self.mirrored = bool(mirrored)
        self.flipped = bool(flipped)
        self.transposed = bool(transposed)
        self.gray = bool(gray)

        self.updateImage()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.updateImage)
        self._timer.setInterval(1000 / self.device.fps)
        self._timer.start()
        self.destroyed.connect(self.stop)

    def stop(self):
        self._timer.stop()
        self.device.stop()

    def close(self):
        self.stop()
        self.device.close()

    @QtCore.pyqtSlot()
    def updateImage(self):
        ready, image = self.device.read()
        if ready:
            image = cv2.cvtColor(image, self._conversion)
            if self.transposed:
                image = cv2.transpose(image)
            if self.flipped or self.mirrored:
                image = cv2.flip(image, self.flipped * (1 - 2 * self.mirrored))
            self.setImage(image, autoLevels=False)
            self.sigNewFrame.emit(image)

    @property
    def paused(self):
        return not self._timer.isActive()

    @paused.setter
    def paused(self, p):
        if p:
            self._timer.stop()
        else:
            self._timer.start()

    @property
    def gray(self):
        if is_cv2():
            return (self._conversion == cv2.cv.CV_BGR2GRAY)
        return (self._conversion == cv2.COLOR_BGR2GRAY)

    @gray.setter
    def gray(self, gray):
        if is_cv2():
            if bool(gray):
                self._conversion = cv2.cv.CV_BGR2GRAY
            else:
                self._conversion = cv2.cv.CV_BGR2RGB
        else:
            if bool(gray):
                self._conversion = cv2.COLOR_BGR2GRAY
            else:
                self._conversion = cv2.COLOR_BGR2RGB


class QVideoWidget(pg.PlotWidget):
    """Demonstration of how to embed a QVideoItem in a display
    widget, illustrating the correct shut-down procedure.
    The embedding widget must call QVideoItem.stop()
    when it closes, otherwise the application will hang.
    """

    def __init__(self, cameraItem=None, **kwargs):
        super(QVideoWidget, self).__init__(**kwargs)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        if cameraItem is None:
            self.camera = QVideoItem(**kwargs)
        else:
            self.camera = cameraItem

        self.addItem(self.camera)
        self.setRange(self.camera.device.roi, padding=0.)
        self.setAspectLocked(True)
        self.setMouseEnabled(x=False, y=False)

    def closeEvent(self, event):
        self.camera.close()


def main():
    import sys
    from PyQt4.QtGui import QApplication

    app = QApplication(sys.argv)
    device = QCameraDevice(gray=True, size=(640, 480))
    item = QVideoItem(device)
    widget = QVideoWidget(item, background='w')
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
