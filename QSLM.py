#!/usr/bin/env python

"""QSLM.py: PyQt abstraction for a Spatial Light Modulator (SLM)."""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np


class QSLM(QtGui.QLabel):

    gray = [QtGui.qRgb(i, i, i) for i in range(256)]

    def __init__(self, parent=None, fake=False, **kwargs):
        super(QSLM, self).__init__(parent)
        desktop = QtGui.QDesktopWidget()
        if (desktop.screenCount() == 2) and not fake:
            rect = desktop.screenGeometry(1)
            self.w, self.h = rect.width(), rect.height()
            parent = desktop.screen(1)
            super(QSLM, self).__init__(parent)
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        else:
            self.w, self.h = 1024, 768
            super(QSLM, self).__init__(parent)
            self.resize(self.w, self.h)
        self.image = QtGui.QImage()
        phi = np.zeros((self.w, self.h), dtype=np.uint8)
        self.data = phi
        self.setData(phi)
        self.show()

    def toImage(self, data):
        img = QtGui.QImage(data.data,
                           data.shape[1], data.shape[0], data.strides[0],
                           QtGui.QImage.Format_Indexed8)
        img.setColorTable(self.gray)
        self.image = img
        return img

    def toPixmap(self, data):
        pixmap = QtGui.QPixmap(self.toImage(data))
        return pixmap

    def setData(self, data):
        self.data = data
        self.setPixmap(self.toPixmap(data))
        self.update()

    def height(self):
        return self.image.height()

    def width(self):
        return self.image.width()


def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    slm = QSLM()
    slm.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
