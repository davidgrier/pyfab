#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QMainWindow
from JansenWidget import Ui_MainWindow
try:
    from jansenlib.video.QSpinnaker.QSpinnaker import QSpinnaker as Camera
except:
    from jansenlib.video.QOpenCV.QOpenCV import QOpenCV as Camera


class Jansen(QMainWindow):

    def __init__(self):
        super(Jansen, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.installCamera(Camera())

    def installCamera(self, camera):
        self.ui.camera.close()
        self.ui.camera = camera
        self.ui.cameraLayout.addWidget(camera)
        self.ui.screen.camera = camera
        self.ui.screen.adjustSize()
        self.ui.filters.screen = self.ui.screen
        self.ui.dvr.source = self.ui.screen.defaultSource
        self.ui.dvr.screen = self.ui.screen


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = Jansen()
    win.show()
    sys.exit(app.exec_())
