#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QSLM.py: PyQt abstraction for a Spatial Light Modulator (SLM)."""

from pyqtgraph.Qt import QtGui, QtCore
import numpy as np


class QSLM(QtGui.QLabel):

    def __init__(self, parent=None, fake=False):
        desktop = QtGui.QDesktopWidget()
        if (desktop.screenCount() == 2) and not fake:
            super(QSLM, self).__init__(desktop.screen(1))
            rect = desktop.screenGeometry(1)
            self.resize(rect.width(), rect.height())
            self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        else:
            super(QSLM, self).__init__(parent)
            w, h = 640, 480
            self.resize(w, h)
            self.setWindowTitle('SLM')
        phi = np.zeros((self.height(), self.width()), dtype=np.uint8)
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
        img = QtGui.QImage(d.data, d.shape[1], d.shape[0], d.strides[0],
                           QtGui.QImage.Format_Indexed8)
        pix = QtGui.QPixmap.fromImage(img)
        self.setPixmap(pix)


def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    slm = QSLM()
    slm.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
