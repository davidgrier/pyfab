#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Jansen is a video-capture GUI intended for holographic video microscopy"""

from PyQt4 import QtGui
from QJansenWidget import QJansenWidget
import sys


class jansen(QtGui.QMainWindow):

    def __init__(self):
        super(jansen, self).__init__()
        self.instrument = QJansenWidget()
        self.init_ui()
        self.show()
        tabs = self.instrument.tabs
        tabs.setFixedWidth(tabs.width())

    def init_ui(self):
        self.setWindowTitle('Jansen')
        self.statusBar().showMessage('Ready')

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')

        exitIcon = QtGui.QIcon.fromTheme('exit')
        exitAction = QtGui.QAction(exitIcon, '&Exit', self)
        exitAction.setShortcut('Ctrl-Q')
        exitAction.setStatusTip('Exit PyFab')
        exitAction.triggered.connect(QtGui.qApp.quit)
        fileMenu.addAction(exitAction)

        self.setCentralWidget(self.instrument)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = jansen()
    sys.exit(app.exec_())
