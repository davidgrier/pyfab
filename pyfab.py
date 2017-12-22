#!/usr/bin/env python

from PyQt4 import QtGui
from QFabWidget import QFabWidget
from objects import fabconfig
import sys


class pyfab(QtGui.QMainWindow):

    def __init__(self):
        super(pyfab, self).__init__()
        self.init_ui()
        self.instrument = QFabWidget()
        self.config = fabconfig(self)
        self.config.restore(self.instrument.wcgh)
        self.setCentralWidget(self.instrument)
        self.show()

    def init_ui(self):
        self.setWindowTitle('PyFab')
        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        fileMenu = menubar.addMenu('&File')
        taskMenu = menubar.addMenu('&Tasks')
        calibrateMenu = menubar.addMenu('&Calibrate')

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
        # exitAction.triggered.connect(QtGui.qApp.quit)
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)

        maxIcon = QtGui.QIcon.fromTheme('document-save')
        maxAction = QtGui.QAction(maxIcon, 'Max Image', self)
        maxAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('maxtask'))
        taskMenu.addAction(maxAction)

        findIcon = QtGui.QIcon.fromTheme('camera-photo')
        findAction = QtGui.QAction(findIcon, 'Find traps', self)
        findAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('findtraps'))
        taskMenu.addAction(findAction)

        rcIcon = QtGui.QIcon.fromTheme('camera-photo')
        rcAction = QtGui.QAction(rcIcon, 'Calibrate rc', self)
        rcAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('calibrate_rc'))
        calibrateMenu.addAction(rcAction)

        cghIcon = QtGui.QIcon.fromTheme('camera-photo')
        cghAction = QtGui.QAction(cghIcon, 'Calibrate CGH', self)
        cghAction.triggered.connect(
            lambda: self.instrument.tasks.registerTask('calibrate_cgh'))
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
