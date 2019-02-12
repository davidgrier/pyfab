#!/usr/bin/env python
# -*- coding: utf-8 -*-

from common.QSettingsWidget import QSettingsWidget
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

    def gray(self):
        return self.device.get('gray')

    def shape(self):
        height = self.device.get('height')
        width = self.device.get('width')
        if self.gray():
            return (height, width)
        else:
            return (height, width, 3)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    device = QSpinnakerThread()
    wid = QSpinnaker(device=device)
    wid.show()
    sys.exit(app.exec_())
