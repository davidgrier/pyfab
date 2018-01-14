#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""pyfab is a GUI for holographic optical trapping"""

from pyqtgraph.Qt import QtGui
from QFabWidget import QFabWidget
from objects import fabconfig
import sys


class pyfab(QtGui.QMainWindow):

    def __init__(self):
        super(pyfab, self).__init__()
        self.instrument = QFabWidget()
        self.config = fabconfig(self)
        self.config.restore(self.instrument.wcgh)
        self.init_ui()
        self.setCentralWidget(self.instrument)
        self.show()
        tabs = self.instrument.tabs
        tabs.setFixedWidth(tabs.width())

    def init_ui(self):
        self.setWindowTitle('PyFab')
        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        taskMenu = menubar.addMenu('&Tasks')
        calibrateMenu = menubar.addMenu('&Calibrate')

        # FILE MENU
        snapIcon = QtGui.QIcon.fromTheme('camera-photo')
        snapAction = QtGui.QAction(snapIcon, 'Save &Photo', self)
        snapAction.setStatusTip('Save a snapshot')
        snapAction.triggered.connect(self.savePhoto)
        fileMenu.addAction(snapAction)

        snapasIcon = QtGui.QIcon.fromTheme('camera-photo')
        snapasAction = QtGui.QAction(snapasIcon, 'Save Photo As ...', self)
        snapasAction.setStatusTip('Save a snapshot')
        snapasAction.triggered.connect(lambda: self.savePhoto(True))
        fileMenu.addAction(snapasAction)

        saveIcon = QtGui.QIcon.fromTheme('document-save')
        saveAction = QtGui.QAction(saveIcon, '&Save Settings', self)
        saveAction.setStatusTip('Save current settings')
        saveAction.triggered.connect(self.saveSettings)
        fileMenu.addAction(saveAction)

        exitIcon = QtGui.QIcon.fromTheme('application-exit')
        exitAction = QtGui.QAction(exitIcon, '&Exit', self)
        exitAction.setShortcut('Ctrl-Q')
        exitAction.setStatusTip('Exit PyFab')
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        # TASK MENU
        clearAction = QtGui.QAction('Clear traps', self)
        clearAction.setStatusTip('Delete all traps')
        clearAction.triggered.connect(self.instrument.pattern.clearTraps)
        taskMenu.addAction(clearAction)

        textAction = QtGui.QAction('Render text', self)
        textAction.setStatusTip('Render text as a pattern of traps')
        textAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('rendertext'))
        taskMenu.addAction(textAction)

        textasAction = QtGui.QAction('Render text ...', self)
        textasAction.setStatusTip('Render text as a pattern of traps')
        textasAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('rendertextas'))
        taskMenu.addAction(textasAction)

        # CALIBRATION MENU
        rcAction = QtGui.QAction('Calibrate rc', self)
        rcAction.setStatusTip('Find location of optical axis in field of view')
        rcAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('calibrate_rc'))
        calibrateMenu.addAction(rcAction)

        cghAction = QtGui.QAction('Aberrations', self)
        cghAction.setStatusTip('NOT IMPLEMENTED YET')
        cghAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('calibrate_haar'))
        calibrateMenu.addAction(cghAction)

    def savePhoto(self, select=False):
        filename = self.config.filename(suffix='.png')
        if select:
            filename = QtGui.QFileDialog.getSaveFileName(
                self, 'Save Snapshot',
                directory=filename,
                filter='Image files (*.png)')
        if filename:
            qimage = self.instrument.fabscreen.video.qimage
            qimage.mirrored(vertical=True).save(filename)
            self.statusBar().showMessage('Saved ' + filename)

    def saveSettings(self):
        self.config.save(self.instrument.wcgh)

    def close(self):
        self.instrument.close()
        QtGui.qApp.quit()

    def closeEvent(self, event):
        self.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = pyfab()
    sys.exit(app.exec_())
