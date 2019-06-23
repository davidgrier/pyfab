#!/usr/bin/env python
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from FabWidget import Ui_PyFab
from common.Configuration import Configuration

from jansenlib.video.QOpenCV.QOpenCV import QOpenCV
from pyfablib.QCGH.CGH import CGH
from pyfablib.QSLM import QSLM

import logging
logging.basicConfig()
logger = logging.getLogger('nujansen')
logger.setLevel(logging.DEBUG)

try:
    from jansenlib.video.QSpinnaker.QSpinnaker import QSpinnaker
except Exception as ex:
    logger.warning(ex)


class Fab(QMainWindow, Ui_PyFab):

    def __init__(self, parent=None, noconfig=False):
        super(Fab, self).__init__(parent)
        self.setupUi(self)
        self.configuration = Configuration(self)

        # camera
        try:
            camera = QSpinnaker()
        except:
            camera = QOpenCV()
        self.installCamera(camera)

        # spatial light modulator
        self.slm = QSLM()

        # computation pipeline
        self.cgh.device = CGH(shape=self.slm.shape)

        # self.cgh.sigHologramReady.connect(self.slm.setData)

        self.configureUi()
        self.connectSignals()

        self.doconfig = not noconfig
        if self.doconfig:
            self.restoreConfiguration()

    def closeEvent(self, event):
        self.saveConfiguration()
        self.screen.close()
        self.deleteLater()

    def installCamera(self, camera):
        self.camera.close()  # remove placeholder widget
        self.camera = camera
        self.cameraLayout.addWidget(camera)
        self.screen.camera = camera

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

    def restoreConfiguration(self):
        if self.doconfig:
            self.configuration.restore(self.camera)

    def saveConfiguration(self):
        if self.doconfig:
            self.configuration.save(self.camera)


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
    win = Fab(noconfig=args.noconfig)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
