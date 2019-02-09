#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2018 David G. Grier and Michael O'Brien
#
# This file is part of pyfab
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.

"""Jansen is a video-capture GUI intended for holographic video microscopy"""

from PyQt5 import QtGui
from pyfab.jansenlib.QJansenWidget import QJansenWidget
from pyfab.common.fabconfig import fabconfig
import sys


class jansen(QtGui.QMainWindow):

    def __init__(self):
        super(jansen, self).__init__()
        self.instrument = QJansenWidget(self)
        self.init_ui()
        self.config = fabconfig(self)
        self.show()

    def init_ui(self):
        self.setWindowTitle('Jansen')
        self.statusBar().showMessage('Ready')
        self.fileMenu()
        self.instrument.setTabBarWidth()
        self.setCentralWidget(self.instrument)

    def fileMenu(self):
        menu = self.menuBar().addMenu('&File')

        icon = QtGui.QIcon.fromTheme('camera-photo')
        action = QtGui.QAction(icon, 'Save &Photo', self)
        action.setShortcut('Ctrl+S')
        action.setStatusTip('Save a snapshot')
        action.triggered.connect(self.savePhoto)
        menu.addAction(action)

        icon = QtGui.QIcon.fromTheme('camera-photo')
        action = QtGui.QAction(icon, 'Save Photo &As ...', self)
        action.setShortcut('Ctrl+A')
        action.setStatusTip('Save a snapshot')
        action.triggered.connect(lambda: self.savePhoto(True))
        menu.addAction(action)

        icon = QtGui.QIcon.fromTheme('application-exit')
        action = QtGui.QAction(icon, '&Quit', self)
        action.setShortcut('Ctrl+Q')
        action.setStatusTip('Quit Jansen')
        action.triggered.connect(self.close)
        menu.addAction(action)

    def savePhoto(self, select=False):
        filename = self.config.filename(suffix='.png')
        if select:
            filename = QtGui.QFileDialog.getSaveFileName(
                self, 'Save Snapshot',
                directory=filename,
                filter='Image files (*.png)')
        if filename:
            qimage = self.instrument.screen.video.qimage
            qimage.mirrored(vertical=True).save(filename)
            self.statusBar().showMessage('Saved ' + filename)

    def close(self):
        self.instrument.close()
        QtGui.qApp.quit()

    def closeEvent(self, event):
        self.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = jansen()
    sys.exit(app.exec_())
