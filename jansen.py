#!/usr/bin/env python
# -*- coding: utf-8 -*-

from PyQt5 import (uic, QtWebEngineWidgets)
from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from PyQt5.QtCore import pyqtSlot
import os

from jansenlib.video import QCamera
from common.Configuration import Configuration

import help.jansen_help_rc

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    ex1 = None
    from jansenlib.QVision import QVision
except Exception as ex:
    ex1 = ex


class Jansen(QMainWindow):

    def __init__(self, parent=None, noconfig=False):
        super(Jansen, self).__init__(parent)

        dir = os.path.dirname(os.path.abspath(__file__))
        uifile = os.path.join(dir, 'jansenlib', 'JansenWidget.ui')
        uic.loadUi(uifile, self)
        
        self.configuration = Configuration(self)

        # Setup vision tab
        try:
            self.vision.close()
            self.vision.setObjectName("vision")
            self.vision = QVision(self.tabVision)
            self.visionLayout.addWidget(self.vision)
            self.setupVision = True
        except Exception as ex2:
            err = ex2 if ex1 is None else ex1
            msg = 'Could not import Machine Vision pipeline: {}'
            logger.warning(msg.format(err))
            self.tabWidget.setTabEnabled(2, False)
            self.setupVision = False

        # Setup camera
        self.camera.close()  # remove placeholder widget from UI
        camera = QCamera()
        self.camera = camera
        self.screen.camera = camera
        self.cameraLayout.addWidget(camera)

        self.configureUi()
        self.connectSignals()

        self.doconfig = not noconfig
        if self.doconfig:
            self.restoreSettings()

    def closeEvent(self, event):
        self.saveSettings()
        self.screen.close()
        self.deleteLater()

    def configureUi(self):
        self.filters.screen = self.screen
        self.histogram.screen = self.screen
        self.dvr.screen = self.screen
        self.dvr.source = self.screen.default
        self.dvr.filename = self.configuration.datadir + 'jansen.avi'
        if self.setupVision:
            self.vision.jansen = self
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
        if self.setupVision:
            self.screen.sigNewFrame.connect(self.vision.process)

    @pyqtSlot()
    def setDvrSource(self, source):
        self.dvr.source = source

    #
    # Slots for menu actions
    #
    def saveImage(self, qimage, select=False):
        if qimage is None:
            return
        if select:
            getname = QFileDialog.getSaveFileName
            filename, _ = getname(self, 'Save Image',
                                  directory=filename,
                                  filter='Image files (*.png)')
        else:
            filename = self.configuration.filename(suffix='.png')
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
            if self.setupVision:
                self.configuration.save(self.vision)

    @pyqtSlot()
    def restoreSettings(self):
        if self.doconfig:
            self.configuration.restore(self.camera)
            if self.setupVision:
                self.configuration.restore(self.vision)


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
