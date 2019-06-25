#!/usr/bin/env python
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from FabWidget import Ui_PyFab
from common.Configuration import Configuration

from jansenlib.video.QOpenCV.QOpenCV import QOpenCV
from pyfablib.QCGH.CGH import CGH
from pyfablib.QSLM import QSLM
from pyfablib.traps.QTrappingPattern import QTrappingPattern

import logging
logging.basicConfig()
logger = logging.getLogger('nujansen')
logger.setLevel(logging.DEBUG)

try:
    from jansenlib.video.QSpinnaker.QSpinnaker import QSpinnaker
except Exception as ex:
    logger.warning(ex)


class Fab(QMainWindow, Ui_PyFab):

    def __init__(self, parent=None, doconfig=True):
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
        self.slm = QSLM(self)

        # computation pipeline
        self.cgh.device = CGH(shape=self.slm.shape)

        # trapping pattern is an interactive overlay
        # that translates user actions into hologram computations
        self.pattern = QTrappingPattern(parent=self)
        self.screen.addOverlay(self.pattern)

        self.configureUi()
        self.connectSignals()

        self.doconfig = doconfig
        if self.doconfig:
            self.restoreConfiguration()

    def closeEvent(self, event):
        self.saveConfiguration()
        self.screen.close()
        self.slm.close()
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
        self.dvr.filename = self.configuration.datadir + 'pyfab.avi'
        self.adjustSize()

    def connectSignals(self):
        self.bcamera.clicked.connect(
            lambda: self.setDvrSource(self.screen.default))
        self.bfilters.clicked.connect(
            lambda: self.setDvrSource(self.screen))
        self.actionSavePhoto.triggered.connect(self.savePhoto)
        self.actionSavePhotoAs.triggered.connect(
            lambda: self.savePhoto(True))

        # 1. Screen events trigger requests for trap updates
        self.screen.sigMousePress.connect(self.pattern.mousePress)
        self.screen.sigMouseRelease.connect(self.pattern.mouseRelease)
        self.screen.sigMouseMove.connect(self.pattern.mouseMove)
        self.screen.sigMouseWheel.connect(self.pattern.mouseWheel)
        # 2. Updates to trapping pattern require hologram calculation
        self.pattern.sigCompute.connect(self.cgh.device.setTraps)
        self.pattern.trapAdded.connect(self.traps.registerTrap)
        # 3. Suppress requests while hologram is being computed
        self.cgh.device.sigComputing.connect(self.screen.pauseSignals)
        # 4. Project result when calculation is complete
        self.cgh.device.sigHologramReady.connect(self.slm.setData)

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
            self.configuration.restore(self.cgh)

    def saveConfiguration(self):
        if self.doconfig:
            self.configuration.save(self.camera)
            self.configuration.save(self.cgh)


def main():
    import sys
    import argparse
    from PyQt5.QtWidgets import QApplication

    parser = argparse.ArgumentParser()
    parser.add_argument('-x', '--noconfig',
                        dest='doconfig', action='store_false',
                        help='Do not use saved configuration data')

    args, unparsed = parser.parse_known_args()
    qt_args = sys.argv[:1] + unparsed

    app = QApplication(qt_args)
    win = Fab(doconfig=args.doconfig)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
