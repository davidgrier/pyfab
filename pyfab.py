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

"""pyfab is a GUI for holographic optical trapping"""

from pyqtgraph.Qt import QtGui
from pyfablib.QFabWidget import QFabWidget
from common.fabconfig import fabconfig
from tasks.taskmenu import taskMenu
import sys


class pyfab(QtGui.QMainWindow):

    def __init__(self):
        self.app = QtGui.QApplication(sys.argv)
        super(pyfab, self).__init__()
        self.instrument = QFabWidget(self)
        self.init_ui()
        self.config = fabconfig(self)
        self.config.restore(self.instrument.wcgh)
        self.show()

    def init_ui(self):
        self.setWindowTitle('PyFab')
        self.statusBar().showMessage('Ready')
        self.menuBar().setNativeMenuBar(False)
        self.fileMenu()
        taskMenu(self)
        self.calibrationMenu()
        self.instrument.setTabBarWidth()
        self.setCentralWidget(self.instrument)

    def fileMenu(self):
        menu = self.menuBar().addMenu('&File')
        icon = QtGui.QIcon.fromTheme('camera-photo')
        action = QtGui.QAction(icon, '&Save Photo', self)
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

        icon = QtGui.QIcon.fromTheme('camera-photo')
        action = QtGui.QAction(icon, 'Save Hologram ...', self)
        action.setStatusTip('Save current hologram')
        action.triggered.connect(lambda: self.saveHologram())
        menu.addAction(action)
        
        icon = QtGui.QIcon.fromTheme('document-save')
        action = QtGui.QAction(icon, 'Save Settings', self)
        action.setShortcut('Ctrl+T')
        action.setStatusTip('Save current settings')
        action.triggered.connect(self.saveSettings)
        menu.addAction(action)

        icon = QtGui.QIcon.fromTheme('application-exit')
        action = QtGui.QAction(icon, '&Quit', self)
        action.setShortcut('Ctrl+Q')
        action.setStatusTip('Quit PyFab')
        action.triggered.connect(self.close)
        menu.addAction(action)

    def calibrationMenu(self):
        menu = self.menuBar().addMenu('&Calibration')
        action = QtGui.QAction('Calibrate rc', self)
        action.setStatusTip('Find location of optical axis in field of view')
        action.triggered.connect(
            lambda: self.instrument.tasks.registerTask('calibrate_rc'))
        menu.addAction(action)

        self.stageMenu(menu)

        action = QtGui.QAction('Aberrations', self)
        action.setStatusTip('NOT IMPLEMENTED YET')
        action.triggered.connect(
            lambda: self.instrument.tasks.registerTask('calibrate_haar'))
        menu.addAction(action)

    def stageMenu(self, parent):
        if self.instrument.wstage is None:
            return
        menu = parent.addMenu('Stage')
        tip = 'Define current position to be stage origin in %s'

        action = QtGui.QAction('Set X origin', self)
        action.setStatusTip(tip % 'X')
        action.triggered.connect(self.instrument.wstage.setXOrigin)
        menu.addAction(action)

        action = QtGui.QAction('Set Y origin', self)
        action.setStatusTip(tip % 'Y')
        action.triggered.connect(self.instrument.wstage.setYOrigin)
        menu.addAction(action)

        action = QtGui.QAction('Set Z origin', self)
        action.setStatusTip(tip % 'Z')
        action.triggered.connect(self.instrument.wstage.setZOrigin)
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

    def saveHologram(self):
        filename = self.config.filename(suffix='.png')
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Save Snapshot',
            directory=filename,
            filter='Image files (*.png)')
        if filename:
            self.instrument.slm.pixmap().save(filename)
            self.statusBar().showMessage('Saved ' + filename)

    def saveSettings(self):
        self.config.save(self.instrument.wcgh)

    def close(self):
        self.instrument.close()
        QtGui.qApp.quit()

    def closeEvent(self, event):
        self.close()


if __name__ == '__main__':
    instrument = pyfab()
    sys.exit(instrument.app.exec_())
