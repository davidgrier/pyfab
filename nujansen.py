#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from JansenWidget import Ui_MainWindow
from common.Configuration import Configuration

import logging
logging.basicConfig()
logger = logging.getLogger('nujansen')
logger.setLevel(logging.DEBUG)

try:
    from jansenlib.video.QSpinnaker.QSpinnaker import QSpinnaker as Camera
except Exception as ex:
    logger.warning(ex)
    from jansenlib.video.QOpenCV.QOpenCV import QOpenCV as Camera


class Jansen(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(Jansen, self).__init__(parent)
        self.setupUi(self)
        self.configuration = Configuration(self)
        self.installCamera(Camera())
        self.configureUi()
        self.connectSignals()

    def closeEvent(self, event):
        self.saveConfiguration()
        self.screen.close()
        self.deleteLater()

    def installCamera(self, camera):
        self.camera.close()  # remove placeholder widget
        self.camera = camera
        self.cameraLayout.addWidget(camera)
        self.screen.camera = camera
        self.configuration.restore(self.camera)

    def configureUi(self):
        self.filters.screen = self.screen
        self.dvr.source = self.screen.defaultSource
        self.dvr.screen = self.screen

    def connectSignals(self):
        self.dvr.recording.connect(self.camera.setDisabled)
        self.dvr.recording.connect(self.filters.setDisabled)
        self.bcamera.clicked.connect(
            lambda: self.setDvrSource(self.screen.defaultSource))
        self.bfilters.clicked.connect(
            lambda: self.setDvrSource(self.screen))
        self.actionSavePhoto.triggered.connect(self.savePhoto)
        self.actionSavePhotoAs.triggered.connect(
            lambda: self.savePhoto(True))

    def setDvrSource(self, source):
        self.dvr.source = source

    def savePhoto(self, select=False):
        filename = self.configuration.filename(suffix='.png')
        if select:
            getname = QFileDialog.getSaveFileName
            filename, _ = getname(self, 'Save Snapshot',
                                  directory=filename,
                                  filter='Image files (*.png)')
        if filename:
            qimage = self.screen.imageItem.qimage
            qimage.mirrored(vertical=True).save(filename)
            self.statusBar().showMessage('Saved ' + filename)

    def saveConfiguration(self):
        self.configuration.save(self.camera)


if __name__ == '__main__':
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    win = Jansen()
    win.show()
    sys.exit(app.exec_())
