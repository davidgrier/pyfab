#!/usr/bin/env python

"""QVideoItem.py: pyqtgraph module for OpenCV video camera."""

import cv2
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
import numpy as np
from QCameraDevice import QCameraDevice


def is_cv2():
    return cv2.__version__.startswith("2.")


class QVideoItem(pg.ImageItem):
    """Video source for pyqtgraph applications.
    Acts like an ImageItem that periodically polls
    a camera for updated video frames.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, device=None, parent=None,
                 mirrored=False,
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
        self._filters = list()

        self._shape = None

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
            if image.ndim == 3:
                image = cv2.cvtColor(image, self._conversion)
            if self.transposed:
                image = cv2.transpose(image)
            if self.flipped or self.mirrored:
                image = cv2.flip(image, self.mirrored * (1 - 2 * self.flipped))
            for filter in self._filters:
                image = filter(image)
            self.setImage(image, autoLevels=False)
            self._shape = image.shape
            self.sigNewFrame.emit(image)

    def shape(self):
        return self._shape

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

    def registerFilter(self, filter):
        self._filters.append(filter)

    def unregisterFilter(self, filter):
        if filter in self._filters:
            self._filters.remove(filter)


class QVideoWidget(pg.PlotWidget):
    """Demonstration of how to embed a QVideoItem in a display
    widget, illustrating the correct shut-down procedure.
    The embedding widget must call QVideoItem.stop()
    when it closes, otherwise the application will hang.
    """

    def __init__(self, cameraItem=None, **kwargs):
        super(QVideoWidget, self).__init__(**kwargs)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)

        if cameraItem is None:
            self.source = QVideoItem(**kwargs)
        else:
            self.source = cameraItem

        self.addItem(self.source)
        self.setRange(self.source.device.roi, padding=0.)
        self.setAspectLocked(True)
        self.setMouseEnabled(x=False, y=False)

    def closeEvent(self, event):
        self.camera.close()


def main():
    import sys
    from PyQt4.QtGui import QApplication

    app = QApplication(sys.argv)
    camera = QCameraDevice(size=(640, 480))
    video = QVideoItem(camera, gray=True)
    widget = QVideoWidget(video, background='w')
    widget.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
