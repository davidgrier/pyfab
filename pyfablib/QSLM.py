#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QSLM.py: PyQt abstraction for a Spatial Light Modulator (SLM)."""

from PyQt5.QtCore import (Qt, pyqtSlot)
from PyQt5.QtWidgets import (QLabel, QDesktopWidget)
from PyQt5.QtGui import (QImage, QPixmap, QGuiApplication)
import numpy as np


class QSLM(QLabel):

    def __init__(self, parent=None, fake=False):
        screens = QGuiApplication.screens()
        if (len(screens) == 2) and not fake:
            super(QSLM, self).__init__(screens[1])
            # super(QSLM, self).__init__(desktop.screen(1))
            # rect = desktop.screenGeometry(1)
            # self.resize(rect.width(), rect.height())
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.showFullScreen()
        else:
            super(QSLM, self).__init__(parent)
            w, h = 640, 480
            self.resize(w, h)
            self.setWindowTitle('SLM')
            self.show()
        phi = np.zeros((self.height(), self.width()), dtype=np.uint8)
        self.data = phi
        # self.show()

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
        img = QImage(d.data, d.shape[1], d.shape[0], d.strides[0],
                     QImage.Format_Indexed8)
        pix = QPixmap.fromImage(img)
        self.setPixmap(pix)


def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    slm = QSLM()
    slm.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
