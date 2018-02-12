#!/usr/bin/env python

"""QSLM.py: PyQt abstraction for a Spatial Light Modulator (SLM)."""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt


class QSLM(QtGui.QLabel):

    def __init__(self, parent=None, fake=False, **kwargs):
        desktop = QtGui.QDesktopWidget()
        if (desktop.screenCount() == 2) and not fake:
            rect = desktop.screenGeometry(1)
            w, h = rect.width(), rect.height()
            parent = desktop.screen(1)
            super(QSLM, self).__init__(parent)
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        else:
            w, h = 1024, 768
            super(QSLM, self).__init__(parent)
            self.resize(w, h)
            self.setWindowTitle('SLM')
        self._width = w
        self._height = h
        phi = np.zeros((h, w), dtype=np.uint8)
        self.data = phi
        self.show()

    @QtCore.pyqtSlot(np.ndarray)
    def setData(self, data):
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        img = QtGui.QImage(ImageQt(Image.fromarray(d)))
        pix = QtGui.QPixmap.fromImage(img)
        self.setPixmap(pix)

    def height(self):
        return self._height

    def width(self):
        return self._width


def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    slm = QSLM()
    slm.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
