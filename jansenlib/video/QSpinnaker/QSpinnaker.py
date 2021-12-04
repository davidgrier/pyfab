#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import uic
from pathlib import Path
from common.QSettingsWidget import QSettingsWidget
from .SpinnakerCamera import SpinnakerCamera


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QSpinnaker(QSettingsWidget):

    '''Camera widget based on Spinnaker SDK'''

    def __init__(self, parent=None, device=None, **kwargs):
        if device is None:
            try:
                device = SpinnakerCamera(**kwargs)
            except IndexError:
                raise IndexError('Cannot connect to camera')
        uifile = Path(__file__).parent.joinpath('QSpinnakerWidget.ui')
        uiclass, _ = uic.loadUiType(uifile)
        super(QSpinnaker, self).__init__(parent=parent,
                                         device=device,
                                         ui=uiclass())
        self.read = self.device.read

    def configureUi(self):
        logger.debug('configuring UI')
        self.ui.framerate.setRange(*self.device.frameraterange)
        self.ui.exposuretime.setRange(*self.device.exposuretimerange)
        self.ui.gain.setRange(*self.device.gainrange)
        self.ui.blacklevel.setRange(*self.device.blacklevelrange)
        self.ui.gamma.setRange(*self.device.gammarange)
        self.ui.x0.setRange(0, self.device.widthmax-10)
        self.ui.y0.setRange(0, self.device.heightmax-10)
        self.ui.width.setRange(10, self.device.widthmax)
        self.ui.height.setRange(10, self.device.heightmax)
        self.widthChanged = self.ui.width.valueChanged
        self.heightChanged = self.ui.height.valueChanged

    def close(self):
        logger.debug('Closing camera interface')
        self.device.close()
        self.device = None

    def closeEvent(self):
        self.close()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    wid = QSpinnaker()
    wid.show()
    sys.exit(app.exec_())
