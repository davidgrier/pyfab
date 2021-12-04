#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import uic
from common.QSettingsWidget import QSettingsWidget
from .OpenCVCamera import OpenCVCamera
from pathlib import Path

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class QOpenCV(QSettingsWidget):

    '''Camera widget based on OpenCV'''

    def __init__(self, parent=None, device=None, **kwargs):
        if device is None:
            device = OpenCVCamera(**kwargs)
        uifile = Path(__file__).parent.joinpath('QOpenCVWidget.ui')
        uiclass, _ = uic.loadUiType(uifile)
        super(QOpenCV, self).__init__(parent,
                                      device=device,
                                      ui=uiclass())
        self.read = self.device.read

    def configureUi(self):
        logger.debug('configuring UI')
        self.ui.width.setMaximum(self.device.width)
        self.ui.height.setMaximum(self.device.height)
        self.widthChanged = self.ui.width.valueChanged
        self.heightChanged = self.ui.height.valueChanged

    def close(self):
        logger.debug('Closing camera interface')
        self.device = None

    def closeEvent(self):
        self.close()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    wid = QOpenCV()
    wid.show()
    sys.edit(app.exec_())
