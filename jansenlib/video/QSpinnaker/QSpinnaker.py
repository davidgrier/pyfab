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
        '''
        self.ui.exposureauto.clicked.connect(
            lambda: self.device.set('exposureauto', 'Once'))
        self.ui.gainauto.clicked.connect(
            lambda: self.device.set('gainauto', 'Once'))
        '''
        pass

    def close(self):
        self.thread.stop()
        self.thread.quit()
        self.thread.wait()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    device = QSpinnakerThread()
    wid = QSpinnaker(device=device)
    wid.show()
    sys.exit(app.exec_())
