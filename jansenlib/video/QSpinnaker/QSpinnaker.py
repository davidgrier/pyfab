#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyfab.common.QSettingsWidget import QSettingsWidget
from QSpinnakerWidget import Ui_QSpinnakerWidget


class QSpinnaker(QSettingsWidget):

    '''Camera widget based on Spinnaker SDK'''

    def __init__(self, parent=None, device=None):
        ui = Ui_QSpinnakerWidget()
        super(QSpinnaker, self).__init__(parent=parent,
                                         device=device,
                                         ui=ui)

    def configureUi(self):
        self.ui.autoexposure.clicked.connect(
            lambda: self.device.set('autoexposure', 'Once'))
        self.ui.autogain.clicked.connect(
            lambda: self.device.set('autogain', 'Once'))
        # limits on widgets


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    from SpinnakerCamera import SpinnakerCamera

    app = QApplication(sys.argv)
    device = SpinnakerCamera()
    wid = QSpinnaker(device=device)
    wid.show()
    sys.exit(app.exec_())
