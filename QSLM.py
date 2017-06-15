#!/usr/bin/env python

"""QSLM.py: PyQt abstraction for a Spatial Light Modulator (SLM)."""

from PyQt4 import QtGui, QtCore
import numpy as np


class QSLM(QtGui.QLabel):

    gray = [QtGui.qRgb(i, i, i) for i in range(256)]

    def __init__(self, parent=None, **kwargs):
        super(QSLM, self).__init__(parent)
        desktop = QtGui.QDesktopWidget()
        if desktop.numScreens() == 2:
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
            rect = desktop.screenGeometry(1)
            self.w, self.h = rect.width(), rect.height()
        else:
            self.w, self.h = 512, 512
        self.image = QtGui.QImage()
        phi = np.zeros((self.w, self.h), dtype=np.uint8)
        self.data = phi
        self.setData(phi)

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
