#!/usr/bin/env python
# -*- coding: utf-8 -*-

from jansenlib.video.QOpenCV.QOpenCV import QOpenCV
from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from PyQt5.QtCore import pyqtSlot

from JansenWidget import Ui_Jansen
from common.Configuration import Configuration

import logging
logging.basicConfig()
logger = logging.getLogger('nujansen^S')
logger.setLevel(logging.DEBUG)

try:
    from jansenlib.QVision.QHVM import QHVM
except Exception as ex:
    logger.warning('Could not import Machine Vision pipeline: {}'.format(ex))

try:
    from jansenlib.video.QSpinnaker.QSpinnaker import QSpinnaker as QCamera
except Exception as ex:
    logger.warning('Could not import Spinnaker camera: {}'.format(ex))
    from jansenlib.video.QOpenCV.QOpenCV import QOpenCV as QCamera


class Jansen(QMainWindow, Ui_Jansen):

    def __init__(self, parent=None, noconfig=False):
        super(Jansen, self).__init__(parent)
        self.setupUi(self)
        self.configuration = Configuration(self)

        # Setup vision tab
        try:
            self.vision.close()
            self.vision.setObjectName("vision")
            self.vision = QHVM(self.tabVision)
            self.visionLayout.addWidget(self.vision)
            self.setupVision = True
        except Exception:
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
        newFrame = self.screen.source.sigNewFrame
        newFrame.connect(self.histogram.updateHistogram)
        if self.setupVision:
            newFrame.connect(self.vision.process)

    @pyqtSlot()
    def setDvrSource(self, source):
        self.dvr.source = source
        if self.setupVision:
            if source is self.screen.default:
                self.screen.source.sigNewFrame.connect(self.vision.process)
                try:
                    self.screen.sigNewFrame.disconnect(self.vision.process)
                except Exception:
                    pass
            else:
                self.screen.sigNewFrame.connect(self.vision.process)
                try:
                    self.screen.source.sigNewFrame.disconnect(
                        self.vision.process)
                except Exception:
                    pass

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
