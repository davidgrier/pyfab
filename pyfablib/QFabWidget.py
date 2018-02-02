#!/usr/bin/env python

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
from jansenlib.QJansenWidget import QJansenWidget, tabLayout
import traps
from proscan.QProscan import QProscan
from IPG.QIPGLaser import QIPGLaser
from QSLM import QSLM
from CGH import CGH, QCGH
import logging
import sys


class hardwareTab(QtGui.QWidget):

    def __init__(self, parent):
        super(hardwareTab, self).__init__(parent=parent)
        self.title = 'Hardware'
        self.index = -1

        layout = tabLayout()
        try:
            self.wstage = QProscan()
            layout.addWidget(self.wstage)
        except ValueError as ex:
            self.wstage = None
            logging.warning('Could not install stage: %s', ex)
        try:
            self.wlaser = QIPGLaser()
            layout.addWidget(self.wlaser)
        except ValueError as ex:
            self.wlaser = None
            logging.warning('Could not install laser: %s', ex)
        self.setLayout(layout)
        self.parent().tabs.currentChanged.connect(self.expose)

    def expose(self, index):
        if index == self.index:
            if self.wstage is not None:
                self.wstage.start()
            if self.wlaser is not None:
                self.wlaser.start()
        else:
            if self.wstage is not None:
                self.wstage.stop()
            if self.wlaser is not None:
                self.wstage.stop()


class QFabWidget(QJansenWidget):

    def __init__(self, **kwargs):
        super(QFabWidget, self).__init__(**kwargs)
        self.init_configuration()

    def init_hardware(self, size):
        super(QFabWidget, self).init_hardware(size)
        # spatial light modulator
        self.slm = QSLM()
        # computation pipeline for the trapping pattern
        self.cgh = CGH(slm=self.slm)
        # self.computethread = QtCore.QThread()
        # self.cgh.moveToThread(self.computethread)
        # self.computethread.start()
        self.wcgh = QCGH(self.cgh, self.screen)
        self.pattern = traps.QTrappingPattern(parent=self.screen,
                                              pipeline=self.cgh)

    def init_ui(self):
        super(QFabWidget, self).init_ui()
        tab = hardwareTab(self)
        index = self.tabs.addTab(tab, 'Hardware')
        tab.index = index
        self.wstage = tab.wstage
        self.tabs.addTab(self.cghTab(), 'CGH')
        self.tabs.addTab(self.trapTab(), 'Traps')

    def cghTab(self):
        wcgh = QtGui.QWidget()
        layout = tabLayout()
        layout.addWidget(self.wcgh)
        wcgh.setLayout(layout)
        return wcgh

    def trapTab(self):
        wtraps = QtGui.QWidget()
        layout = tabLayout()
        layout.addWidget(traps.QTrapWidget(self.pattern))
        wtraps.setLayout(layout)
        return wtraps

    def init_configuration(self):
        sz = self.screen.video.device.size
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
