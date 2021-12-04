#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import uic
from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from PyQt5.QtCore import pyqtSlot

from jansenlib.video import QCamera
from common.Configuration import Configuration

# Support for HTML help system
from PyQt5 import QtWebEngineWidgets
import help.jansen_help_rc

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Jansen(QMainWindow):

    def __init__(self, parent=None, noconfig=False):
        super(Jansen, self).__init__(parent)

        uifile = Path('jansenlib').joinpath('JansenWidget.ui')
        uic.loadUi(uifile, self)

        self.configuration = Configuration(self)

        self.setupCamera()
        self.configureUi()
        self.connectSignals()

        self.doconfig = not noconfig
        if self.doconfig:
            self.restoreSettings()

    def closeEvent(self, event):
        self.saveSettings()
        self.screen.close()
        self.deleteLater()

    def setupCamera(self):
        self.camera.close()  # remove placeholder widget from UI
        camera = QCamera()
        self.camera = camera
        self.screen.camera = camera
        self.cameraLayout.addWidget(camera)

    def configureUi(self):
        self.filters.screen = self.screen
        self.histogram.screen = self.screen
        self.dvr.screen = self.screen
        self.dvr.source = self.screen.default
        self.dvr.filename = self.configuration.datadir + 'jansen.avi'
        self.adjustSize()

    def connectSignals(self):
        self.bcamera.clicked.connect(
            lambda: self.setDvrSource(self.screen.default))
        self.bfilters.clicked.connect(
            lambda: self.setDvrSource(self.screen))
        self.actionSavePhoto.triggered.connect(self.savePhoto)
        self.actionSavePhotoAs.triggered.connect(
            lambda: self.savePhoto(True))

        # Signals associated with handling images
        self.screen.source.sigNewFrame.connect(self.histogram.updateHistogram)

    @pyqtSlot()
    def setDvrSource(self, source):
        self.dvr.source = source

    #
    # Slots for menu actions
    #
    def saveImage(self, qimage, select=False):
        if qimage is None:
            return
        filename = self.configuration.filename(suffix='.png')
        if select:
            getname = QFileDialog.getSaveFileName
            filename, _ = getname(self, 'Save Image',
                                  directory=filename,
                                  filter='Image files (*.png)')
        if filename:
            qimage.save(filename)
            self.statusBar().showMessage('Saved ' + filename)

    @pyqtSlot()
    def savePhoto(self, select=False):
        qimage = self.screen.imageItem.qimage.mirrored(vertical=True)
        self.saveImage(qimage, select=select)

    @pyqtSlot()
    def savePhotoAs(self):
        self.savePhoto(select=True)

    @pyqtSlot()
    def saveSettings(self):
        if self.doconfig:
            self.configuration.save(self.camera)

    @pyqtSlot()
    def restoreSettings(self):
        if self.doconfig:
            self.configuration.restore(self.camera)


def main():
    import sys
    import argparse
    from PyQt5.QtWidgets import QApplication

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--noconfig',
                        dest='noconfig', action='store_true',
                        help='Do not use saved configuration data')

    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    app = QApplication(qt_args)
    win = Jansen(noconfig=args.noconfig)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
