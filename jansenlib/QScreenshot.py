# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot)
from PyQt5.QtGui import QPixmap
import numpy as np


class QScreenshot(QObject):

    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self, parent, widget=None):
        super(QScreenshot, self).__init__(parent)
        self.window = widget.winId()
        self.gray = False

    @pyqtSlot(np.ndarray)
    def takeScreenshot(self, frame=None):
        pixmap = QPixmap().grabWindow(self.window)
        image = pixmap.toImage()
        self.shape = (image.height(), image.width(), 4)
        buf = image.bits().asstring(image.numBytes())
        result = np.frombuffer(buf, np.uint8).reshape(self.shape)
        self.sigNewFrame.emit(result[:, :, 0:3])

    def height(self):
        return self.shape[0]

    def width(self):
        return self.shape[1]
