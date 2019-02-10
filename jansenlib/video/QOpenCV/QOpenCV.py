# -*- coding: utf-8 -*-

from common.QSettingsWidget import QSettingsWidget
from QOpenCVWidget import Ui_QOpenCVWidget
from QOpenCVThread import QOpenCVThread


class QOpenCV(QSettingsWidget):

    '''Camera widget based on OpenCV'''

    def __init__(self, parent=None, device=None, **kwargs):
        ui = Ui_QOpenCVWidget()
        if device is None:
            device = QOpenCVThread(**kwargs)
        self.sigNewFrame = device.sigNewFrame
        self.start = device.start
        super(QOpenCV, self).__init__(parent,
                                      device=device,
                                      ui=ui)

    def configureUi(self):
        self.ui.width.setMaximum(self.device.width)
        self.ui.height.setMaximum(self.device.height)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    device = QOpenCVThread()
    wid = QOpenCV(device=device)
    wid.show()
    sys.exit(app.exec_())
