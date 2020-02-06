#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import (QMainWindow, QFileDialog)
from PyQt5.QtCore import pyqtSlot

from FabWidget import Ui_PyFab

from pyfablib.QSLM import QSLM
from pyfablib.traps.QTrappingPattern import QTrappingPattern
from tasks.taskmenu import buildTaskMenu
from tasks.Taskmanager import Taskmanager
from common.Configuration import Configuration

import logging
logging.basicConfig()
logger = logging.getLogger('pyfab')
logger.setLevel(logging.DEBUG)

try:
    ex1 = None
    from jansenlib.QVision import QVision
except Exception as ex:
    ex1 = ex

try:
    from pyfablib.QCGH.cupyCGH import cupyCGH as CGH
except Exception as ex:
    logger.warning('Could not import GPU pipeline: {}'.format(ex))
    from pyfablib.QCGH.CGH import CGH

try:
    from jansenlib.video.QSpinnaker.QSpinnaker import QSpinnaker as QCamera
except Exception as ex:
    logger.warning('Could not import Spinnaker camera: {}'.format(ex))
    from jansenlib.video.QOpenCV.QOpenCV import QOpenCV as QCamera


class PyFab(QMainWindow, Ui_PyFab):

    def __init__(self, parent=None, doconfig=True):
        super(PyFab, self).__init__(parent)

        self.setupUi(self)
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

        # camera
        self.camera.close()  # remove placeholder widget from UI
        camera = QCamera()
        self.camera = camera
        self.screen.camera = camera
        self.cameraLayout.addWidget(camera)

        # spatial light modulator
        self.slm = QSLM(self)

        # computation pipeline
        self.cgh.device = CGH(self, shape=self.slm.shape)
        self.cgh.device.start()

        # trapping pattern is an interactive overlay
        # that translates user actions into hologram computations
        self.pattern = QTrappingPattern(parent=self)
        self.screen.addOverlay(self.pattern)

        # process automation
        self.tasks = Taskmanager(self)

        self.configureUi()
        self.connectSignals()

        self.doconfig = doconfig
        if self.doconfig:
            self.restoreSettings()

    def closeEvent(self, event):
        self.saveSettings()
        self.pattern.clearTraps()
        self.screen.close()
        self.slm.close()
        self.cgh.device.stop()
        self.deleteLater()

    def configureUi(self):
        self.filters.screen = self.screen
        self.histogram.screen = self.screen
        self.dvr.screen = self.screen
        self.dvr.source = self.screen.default
        self.dvr.filename = self.configuration.datadir + 'pyfab.avi'
        if self.setupVision:
            self.vision.jansen = self
        index = 2
        self.hardware.index = index
        self.tabWidget.currentChanged.connect(self.hardware.expose)
        self.tabWidget.setTabEnabled(index, self.hardware.has_content())
        self.slmView.setRange(xRange=[0, self.slm.width()],
                              yRange=[0, self.slm.height()],
                              padding=0)
        self.slmView.setYRange(0, self.slm.height())
        buildTaskMenu(self)
        self.adjustSize()

    def connectSignals(self):
        # Signals associated with GUI controls
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

        # Signals associated with the CGH pipeline
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
        self.cgh.device.sigHologramReady.connect(self.slmView.setData)

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
    def saveHologram(self, select=False):
        self.saveImage(self.slm.qimage, select=select)

    @pyqtSlot()
    def saveHologramAs(self):
        self.saveHologram(select=True)

    @pyqtSlot()
    def saveSettings(self):
        if self.doconfig:
            self.configuration.save(self.camera)
            self.configuration.save(self.cgh)

    @pyqtSlot()
    def restoreSettings(self):
        if self.doconfig:
            self.configuration.restore(self.camera)
            self.configuration.restore(self.cgh)

    @pyqtSlot()
    def pauseTasks(self):
        self.tasks.pauseTasks()
        msg = 'Tasks paused' if self.tasks.paused() else 'Tasks running'
        self.statusBar().showMessage(msg)

    @pyqtSlot()
    def stopTasks(self):
        self.tasks.clearTasks()
        self.statusBar().showMessage('Task queue cleared')


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
    win = PyFab(doconfig=args.doconfig)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
