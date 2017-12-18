from PyQt4 import QtGui
from QFabWidget import QFabWidget
from objects import fabconfig
import sys
import os


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

        snapIcon = QtGui.QIcon.fromTheme('camera-photo')
        snapAction = QtGui.QAction(snapIcon, 'Save &Photo', self)
        snapAction.setStatusTip('Save a snapshot')
        snapAction.triggered.connect(self.savePhoto)
        fileMenu.addAction(snapAction)

        saveIcon = QtGui.QIcon.fromTheme('document-save')
        saveAction = QtGui.QAction(saveIcon, '&Save Settings', self)
        saveAction.setStatusTip('Save current settings')
        saveAction.triggered.connect(self.saveSettings)
        fileMenu.addAction(saveAction)

        exitIcon = QtGui.QIcon.fromTheme('application-exit')
        exitAction = QtGui.QAction(exitIcon, '&Exit', self)
        exitAction.setShortcut('Ctrl-Q')
        exitAction.setStatusTip('Exit PyFab')
        exitAction.triggered.connect(QtGui.qApp.quit)
        fileMenu.addAction(exitAction)

    def savePhoto(self):
        filename, filter = QtGui.QFileDialog.getSaveFileName(
            parent=self, caption='Save Snapshot',
            directory=os.path.expanduser('~/data'),
            filter='*.png')
        print(filename)

    def saveSettings(self):
        self.config.save(self.instrument.wcgh)

    def closeEvent(self, event):
        self.config.query_save(self.instrument.wcgh)
        self.instrument.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = pyfab()
    sys.exit(app.exec_())
