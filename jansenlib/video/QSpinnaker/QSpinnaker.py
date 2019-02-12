#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtProperty
from pyfab.common.QSettingsWidget import QSettingsWidget
from QSpinnakerWidget import Ui_QSpinnakerWidget
from QSpinnakerThread import QSpinnakerThread


class QSpinnaker(QSettingsWidget):

    '''Camera widget based on Spinnaker SDK'''

    def __init__(self, parent=None, device=None, **kwargs):
        if device is None:
            device = QSpinnakerThread(**kwargs)
        print(device)
        self.sigNewFrame = device.sigNewFrame
        ui = Ui_QSpinnakerWidget()
        super(QSpinnaker, self).__init__(parent=parent,
                                         device=device,
                                         ui=ui)

    def configureUi(self):
        self.ui.exposureauto.clicked.connect(
            lambda: self.device.set('exposureauto', 'Once'))
        self.ui.gainauto.clicked.connect(
            lambda: self.device.set('gainauto', 'Once'))
        # limits on widgets

    @pyqtProperty(object)
    def shape(self):
        if self.device.gray:
            return(self.device.height, self.device.width)
        return (self.device.height, self.device.width, 3)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys
    from QSpinnakerThread import QSpinnakerThread

    app = QApplication(sys.argv)
    device = QSpinnakerThread()
    wid = QSpinnaker(device=device)
    wid.show()
    sys.exit(app.exec_())
