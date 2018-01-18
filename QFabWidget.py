#!/usr/bin/env python

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui, QtCore
from QJansenWidget import QJansenWidget
import traps
import objects
import sys
import logging


class QFabWidget(QJansenWidget):

    def __init__(self, **kwargs):
        super(QFabWidget, self).__init__(**kwargs)
        self.init_configuration()

    def init_hardware(self, size):
        super(QFabWidget, self).init_hardware(size)
        # spatial light modulator
        self.slm = objects.QSLM()
        # computation pipeline for the trapping pattern
        try:
            self.cgh = objects.cudaCGH(slm=self.slm)
        except (NameError, AttributeError) as ex:
            logging.warning('could not load cudaCGH: %s', ex)
            self.cgh = objects.CGH(slm=self.slm)
        # self.computethread = QtCore.QThread()
        # self.cgh.moveToThread(self.computethread)
        # self.computethread.start()
        self.wcgh = objects.QCGH(self.cgh, self.fabscreen)
        self.pattern = traps.QTrappingPattern(gui=self.fabscreen,
                                              pipeline=self.cgh)

    def init_ui(self):
        super(QFabWidget, self).init_ui()
        self.tabs.addTab(self.hardwareTab(), 'Hardware')
        self.tabs.addTab(self.cghTab(), 'CGH')
        self.tabs.addTab(self.trapTab(), 'Traps')

    def tabLayout(self):
        layout = QtGui.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(1)
        return layout

    def hardwareTab(self):
        whard = QtGui.QWidget()
        layout = self.tabLayout()
        try:
            self.wstage = objects.QProscan()
            layout.addWidget(self.wstage)
        except ValueError as ex:
            self.wstage = None
            logging.warning('Could not install stage: %s', ex)
        try:
            self.wtrappinglaser = objects.QIPGLaser()
            layout.addWidget(self.wtrappinglaser)
        except ValueError as ex:
            self.wtrappinglaser = None
            logging.warning('Could not install laser: %s', ex)
        whard.setLayout(layout)
        return whard

    def cghTab(self):
        wcgh = QtGui.QWidget()
        layout = self.tabLayout()
        layout.addWidget(self.wcgh)
        wcgh.setLayout(layout)
        return wcgh

    def trapTab(self):
        wtraps = QtGui.QWidget()
        layout = self.tabLayout()
        layout.addWidget(traps.QTrapWidget(self.pattern))
        wtraps.setLayout(layout)
        return wtraps

    def init_configuration(self):
        sz = self.fabscreen.video.device.size
        self.wcgh.xc = sz.width() / 2
        self.wcgh.yc = sz.height() / 2
        self.wcgh.zc = 0.
        sz = self.slm.size()
        self.wcgh.xs = sz.width() / 2
        self.wcgh.ys = sz.height() / 2

    def close(self):
        self.pattern.clearTraps()
        self.slm.close()

    def closeEvent(self, event):
        self.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QFabWidget()
    sys.exit(app.exec_())
