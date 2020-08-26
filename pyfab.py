#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from PyQt5.QtWidgets import (QMainWindow, QFileDialog, QStackedLayout)
from PyQt5.QtCore import pyqtSlot

from FabWidget import Ui_PyFab

from jansenlib.video import QCamera
from pyfablib.QCGH import CGH
from pyfablib.QSLM import QSLM
from pyfablib.traps import QTrappingPattern
from tasks import (buildTaskMenu, QTaskmanager)
from common.Configuration import Configuration

from tasks.QVision import QVision

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# # NOTE: How is QVision related to standard set of objects?
# try:
#     ex1 = None
#     from jansenlib.QVision.QHVM import QHVM as QVision
# except Exception as ex:
#     ex1 = ex


class PyFab(QMainWindow, Ui_PyFab):

    def __init__(self, parent=None, doconfig=True):
        super(PyFab, self).__init__(parent)

        self.setupUi(self)
        self.configuration = Configuration(self)

        # Camera
        self.camera.close()  # remove placeholder widget from UI
        self.camera = QCamera()
        self.screen.camera = self.camera
        self.cameraLayout.addWidget(self.camera)        
       

        # Spatial light modulator
        self.slm = QSLM(self)

        # Computation pipeline
        self.cgh.device = CGH(self, shape=self.slm.shape).start()

        # Trapping pattern is an interactive overlay
        # that translates user actions into hologram computations
        self.pattern = QTrappingPattern(parent=self)
        self.screen.addOverlay(self.pattern)

        # Process automation
        self.tasks = QTaskmanager(self)
        self.TaskManagerView.setModel(self.tasks)

#         Setup vision tab
#        try:
#            self.vision.close()
#            self.vision.setObjectName("vision")
#            self.vision = QVision(parent=self.tabVision, pyfab=self)
#            self.visionLayout.addWidget(self.vision)
#            self.tabWidget.setTabEnabled(2, True)
#            self.setupVision = True
#        except Exception as ex2:
#            err = ex2 if ex1 is None else ex1
#            msg = 'Could not import Machine Vision pipeline: {}'
#            logger.warning(msg.format(err))
#            self.tabWidget.setTabEnabled(2, False)
#            self.setupVision = False

        self.tabWidget.setTabEnabled(2, False)#            
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
#         if self.setupVision:
#             self.vision.jansen = self
        self.TaskPropertiesLayout = QStackedLayout(self.TaskPropertiesView)
        index = 4
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
#         self.tasks.dataChanged.connect(lambda x: print('Data Changed!'))
        self.bcamera.clicked.connect(
            lambda: self.setDvrSource(self.screen.default))
        self.bfilters.clicked.connect(
            lambda: self.setDvrSource(self.screen))
        self.bpausequeue.clicked.connect(self.pauseTasks)
        self.bclearqueue.clicked.connect(self.stopTasks)
        
        self.TaskManagerView.clicked.connect(self.tasks.displayProperties)
        self.TaskManagerView.doubleClicked.connect(self.tasks.toggleSelected)

        # Signals associated with handling images
        newframe = self.screen.source.sigNewFrame
        newframe.connect(self.histogram.updateHistogram)
#         if self.setupVision:
#             newframeFrame.connect(self.vision.process)

        # Signals associated with the CGH pipeline
        # 1. Screen events trigger requests for trap updates
        self.screen.sigMousePress.connect(self.pattern.mousePress)
        self.screen.sigMouseRelease.connect(self.pattern.mouseRelease)
        self.screen.sigMouseMove.connect(self.pattern.mouseMove)
        self.screen.sigMouseWheel.connect(self.pattern.mouseWheel)
        # 2. Trap widget reflects changes to trapping pattern
        self.pattern.sigCompute.connect(self.cgh.device.compute)
        self.pattern.trapAdded.connect(self.traps.registerTrap)
        # 3. Project result when calculation is complete
        self.cgh.device.sigHologramReady.connect(self.slm.setData)
        self.cgh.device.sigHologramReady.connect(self.slmView.setData)

        # CGH computations are coordinated with camera
        newframe.connect(self.pattern.refresh)

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
            directory = self.configuration.datadir
            filename, _ = getname(self, 'Save Image',
                                  directory=directory,
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
#             if self.setupVision:
#                 self.configuration.save(self.vision)

    @pyqtSlot()
    def restoreSettings(self):
        if self.doconfig:
            self.configuration.restore(self.camera)
            self.configuration.restore(self.cgh)
#             if self.setupVision:
#                 self.configuration.restore(self.vision)

    @pyqtSlot()
    def pauseTasks(self):
        self.tasks.pauseTasks()
        msg = 'Tasks paused' if self.tasks.paused else 'Tasks running'
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
