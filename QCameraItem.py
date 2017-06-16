#!/usr/bin/env python

"""QCameraItem.py: pyqtgraph module for OpenCV video camera."""

import cv2
import pyqtgraph as pg
from pyqtgraph import QtCore
from PyQt4.QtCore import Qt
import numpy as np


def is_cv2():
    return cv2.__version__.startswith("2.")


class QCameraThread(QtCore.QThread):
    """Grab frames as fast as possible to minimize
    latency for frame acquisition.
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
                 mirrored=True,
                 flipped=True,
                 transposed=True,
                 gray=False,
                 size=None,
                 parent=None):
        super(QCameraDevice, self).__init__(parent)

        self.mirrored = mirrored
        self.flipped = flipped
        self.transposed = transposed
        self.gray = gray

        self.camera = cv2.VideoCapture(cameraId)
        self.thread = QCameraThread(self.camera)

        self.size = size
        # self.fps = int(self.camera.get(cv2.CAP_PROP_FPS))
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
        if ready:
            frame = cv2.cvtColor(frame, self._conversion)
            if self.transposed:
                frame = cv2.transpose(frame)
            if self.flipped or self.mirrored:
                frame = cv2.flip(frame, self.flipped*(1-2*self.mirrored))
        return ready, frame

    @property
    def size(self):
        if is_cv2():
            h = long(self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
            w = long(self.camera.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
        else:
            h = long(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = long(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
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

    @property
    def mirrored(self):
        return self._mirrored

    @mirrored.setter
    def mirrored(self, mirrored):
        self._mirrored = bool(mirrored)

    @property
    def flipped(self):
        return self._flipped

    @flipped.setter
    def flipped(self, flipped):
        self._flipped = bool(flipped)

    @property
    def transposed(self):
        return self._transposed

    @transposed.setter
    def transposed(self, transposed):
        self._transposed = bool(transposed)

    @property
    def roi(self):
        return QtCore.QRectF(0., 0., self.size.width(), self.size.height())


class QCameraItem(pg.ImageItem):
    """Video source for pyqtgraph applications.
    Acts like an ImageItem that periodically polls
    a camera for updated video frames.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)
    
    def __init__(self, cameraDevice=None, parent=None, **kwargs):
        super(QCameraItem, self).__init__(parent, **kwargs)

        if cameraDevice is None:
            self.cameraDevice = QCameraDevice(**kwargs).start()
        else:
            self.cameraDevice = cameraDevice.start()
        self.updateImage()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.updateImage)
        self._timer.setInterval(1000 / self.cameraDevice.fps)
        self._timer.start()
        self.destroyed.connect(self.stop)

    def stop(self):
        self._timer.stop()
        self.cameraDevice.stop()

    def close(self):
        self.stop()
        self.cameraDevice.close()

    @QtCore.pyqtSlot()
    def updateImage(self):
        ready, image = self.cameraDevice.read()
        if ready:
            self._image = image.copy()
            self.setImage(self._image, autoLevels=False)
            self.sigNewFrame.emit(self._image)

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
    def size(self):
        return self.cameraDevice.size

    @size.setter
    def size(self, s):
        pass

    @property
    def roi(self):
        return self.cameraDevice.roi

    @roi.setter
    def roi(self, r):
        pass


class QCameraWidget(pg.PlotWidget):
    """Demonstration of how to embed a QCameraItem in a display
    widget, illustrating the correct shut-down procedure.
    The embedding widget must call QCameraItem.stop()
    when it closes, otherwise the application will hang.
    """

    def __init__(self, cameraItem=None, **kwargs):
        super(QCameraWidget, self).__init__(**kwargs)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        if cameraItem is None:
            self.cameraItem = QCameraItem(**kwargs)
        else:
            self.cameraItem = cameraItem

        self.addItem(self.cameraItem)
        self.setRange(self.cameraItem.roi, padding=0.)
        self.setAspectLocked()
        self.setMouseEnabled(x=False, y=False)

    def closeEvent(self, event):
        self.cameraItem.close()


def main():
    import sys
    from PyQt4.QtGui import QApplication

    app = QApplication(sys.argv)
    device = QCameraDevice(gray=True, size=(640, 480))
    item = QCameraItem(device)
    widget = QCameraWidget(item, background='w')
    widget.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
