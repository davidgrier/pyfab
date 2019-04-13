#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QMainWindow
from JansenWidget import Ui_MainWindow
import logging
logging.basicConfig()
logger = logging.getLogger('nujansen')
logger.setLevel(logging.DEBUG)

try:
    from jansenlib.video.QSpinnaker.QSpinnaker import QSpinnaker as Camera
except Exception as ex:
    logger.warning(ex)
    from jansenlib.video.QOpenCV.QOpenCV import QOpenCV as Camera


class Jansen(QMainWindow):

    def __init__(self):
        super(Jansen, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.installCamera(Camera())
        self.configureUi()
        self.connectSignals()

    def installCamera(self, camera):
        self.ui.camera.close()
        self.ui.camera = camera
        self.ui.cameraLayout.addWidget(camera)
        self.ui.screen.camera = camera

    def configureUi(self):
        self.ui.filters.screen = self.ui.screen
        self.ui.dvr.source = self.ui.screen.defaultSource
        self.ui.dvr.screen = self.ui.screen

    def connectSignals(self):
        self.ui.dvr.recording.connect(self.ui.camera.setDisabled)
        self.ui.dvr.recording.connect(self.ui.filters.setDisabled)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = Jansen()
    win.show()
    sys.exit(app.exec_())
