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
        exitIcon = QtGui.QIcon.fromTheme('exit')
        exitAction = QtGui.QAction(exitIcon, '&Exit', self)
        exitAction.setShortcut('Ctrl-Q')
        exitAction.setStatusTip('Exit PyFab')
        exitAction.triggered.connect(QtGui.qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

    def closeEvent(self, event):
        self.config.query_save(self.instrument.wcgh)
        self.instrument.close()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = pyfab()
    sys.exit(app.exec_())
