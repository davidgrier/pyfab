from PyQt4 import QtGui
from pyfab import pyfab
import sys


class qpyfab(QtGui.QMainWindow):
    def __init__(self):
        super(qpyfab, self).__init__()
        self.instrument = pyfab()
        self.init_ui()

    def init_ui(self):
        self.statusBar().showMessage('Ready')
        self.setWindowTitle('PyFab')

        exitIcon = QtGui.QIcon.fromTheme('exit')
        exitAction = QtGui.QAction(exitIcon, '&Exit', self)
        exitAction.setShortcut('Ctrl-Q')
        exitAction.setStatusTip('Exit PyFab')
        exitAction.triggered.connect(QtGui.qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)

        self.setCentralWidget(self.instrument)
        self.show()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    qpyfab()
    sys.exit(app.exec_())
