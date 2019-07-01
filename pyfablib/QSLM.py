#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QSLM.py: PyQt abstraction for a Spatial Light Modulator (SLM)."""

from PyQt5.QtCore import (Qt, pyqtSlot)
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import (QMainWindow, QImage, QPixmap, QGuiApplication)
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QSLM(QMainWindow):

    def __init__(self, parent=None, fake=False):
        super(QSLM, self).__init__(None, Qt.FramelessWindowHint)
        self.label = QLabel(self)
        self.setCentralWidget(self.label)
        screens = QGuiApplication.screens()
        if (len(screens) == 2) and not fake:
            logger.debug('Opening SLM on secondary screen')
            screen = screens[1]
            geometry = screen.availableGeometry()
            self.setGeometry(geometry)
            self.show()
            self.windowHandle().setScreen(screen)
            self.showFullScreen()
        else:
            w, h = 640, 480
            self.setGeometry(100, 100, w, h)
            self.label.resize(w, h)
            self.show()
        logger.debug(self.size())
        phi = np.zeros((self.height(), self.width()), dtype=np.uint8)
        self.data = phi

    @property
    def shape(self):
        return (self.height(), self.width())

    @pyqtSlot(np.ndarray)
    def setData(self, data):
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        self.qimage = QImage(d.data,
                             d.shape[1], d.shape[0], d.strides[0],
                             QImage.Format_Indexed8)
        self.label.setPixmap(QPixmap.fromImage(self.qimage))


def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    slm = QSLM()
    slm.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
