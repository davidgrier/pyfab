#!/usr/bin/env python

"""QSLM.py: PyQt abstraction for a Spatial Light Modulator (SLM)."""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np


class QSLM(QtGui.QLabel):

    gray = [QtGui.qRgb(i, i, i) for i in range(256)]

    def __init__(self, parent=None, fake=False, **kwargs):
        desktop = QtGui.QDesktopWidget()
        if (desktop.screenCount() == 2) and not fake:
            rect = desktop.screenGeometry(1)
            w, h = rect.width(), rect.height()
            parent = desktop.screen(1)
            super(QSLM, self).__init__(parent)
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        else:
            w, h = 640, 480
            super(QSLM, self).__init__(parent)
            self.resize(w, h)
            self.setWindowTitle('SLM')
        self._width = w
        self._height = h
        phi = np.zeros((h, w), dtype=np.uint8)
        self.data = phi
        self.show()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d
        print(d.shape,d.strides)
        img = QtGui.QImage(d.data,
                           d.shape[0], d.shape[1], d.shape[0],
                           QtGui.QImage.Format_Indexed8)
        img.setColorTable(self.gray)
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
