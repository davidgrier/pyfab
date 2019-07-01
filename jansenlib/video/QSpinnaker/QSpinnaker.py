#!/usr/bin/env python
# -*- coding: utf-8 -*-

from common.QSettingsWidget import QSettingsWidget
from .QSpinnakerWidget import Ui_QSpinnakerWidget
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
        ui = Ui_QSpinnakerWidget()
        super(QSpinnaker, self).__init__(parent=parent,
                                         device=device,
                                         ui=ui)
        self.read = self.device.read

    def configureUi(self):
        logger.debug('configuring UI')
        self.ui.exposure.setRange(self.device.exposuremin,
                                  self.device.exposuremax)
        self.ui.gain.setRange(self.device.gainmin, self.device.gainmax)

    def close(self):
        logger.debug('Closing camera interface')
        self.device.close()
        self.device = None

    def closeEvent(self):
        self.close()

    @property
    def shape(self):
        return self.device.shape


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    wid = QSpinnaker()
    wid.show()
    sys.exit(app.exec_())
