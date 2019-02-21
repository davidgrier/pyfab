#!/usr/bin/env python
# -*- coding: utf-8 -*-

from common.QSettingsWidget import QSettingsWidget
from QSpinnakerThread import QSpinnakerThread
from QSpinnakerWidget import Ui_QSpinnakerWidget

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QSpinnaker(QSettingsWidget):

    '''Camera widget based on Spinnaker SDK'''

    def __init__(self, parent=None, device=None, **kwargs):
        if device is None:
            device = QSpinnakerThread(**kwargs)
        self.thread = device
        self.sigNewFrame = self.thread.sigNewFrame
        ui = Ui_QSpinnakerWidget()
        super(QSpinnaker, self).__init__(parent=parent,
                                         device=device.camera,
                                         ui=ui)
        self.thread.start()

    def configureUi(self):
        logger.debug('configuring UI')
        self.ui.exposure.setRange(self.device.exposuremin,
                                  self.device.exposuremax)
        self.ui.gain.setRange(self.device.gainmin, self.device.gainmax)

    def close(self):
        logger.debug('Closing camera interface')
        # remove reference to camera device so that it can close
        self.device = None
        # shut down acquisition thread
        self.thread.stop()
        self.thread.quit()
        self.thread.wait()

    def closeEvent(self):
        self.close()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    device = QSpinnakerThread()
    wid = QSpinnaker(device=device)
    wid.show()
    sys.exit(app.exec_())
