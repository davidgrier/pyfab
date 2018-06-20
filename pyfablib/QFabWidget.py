#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui, QtCore
from jansenlib.QJansenWidget import QJansenWidget
from .QHardwareTab import QHardwareTab
from .QSLMTab import QSLMTab
from common.tabLayout import tabLayout
from .traps import *
from .QSLM import QSLM
from .CGH import CGH, QCGHPropertyWidget
import sys


class QFabWidget(QJansenWidget):

    def __init__(self, **kwargs):
        super(QFabWidget, self).__init__(**kwargs)

    def init_components(self):
        super(QFabWidget, self).init_components()
        # spatial light modulator
        self.slm = QSLM()
        # computation pipeline for the trapping pattern
        self.cgh = CGH(slm=self.slm)
        self.cgh.sigHologramReady.connect(self.slm.setData)
        self.wcgh = QCGHPropertyWidget(self)
        # trapping pattern is an interactive overlay
        # that translates user actions into hologram computations
        self.pattern = QTrappingPattern(parent=self.screen)
        self.pattern.sigCompute.connect(self.cgh.setTraps)
        self.cgh.sigComputing.connect(self.pattern.pauseSignals)

        self.thread = QtCore.QThread()
        self.thread.start()
        self.cgh.moveToThread(self.thread)
        self.thread.started.connect(self.cgh.start)
        self.thread.finished.connect(self.cgh.stop)

    def init_ui(self):
        super(QFabWidget, self).init_ui()
        # Help tab is at last index
        help_index = self.tabs.count() - 1
        # add new tabs
        hwtab = QHardwareTab()
        if hwtab.has_content():
            self.tabs.addTab(hwtab, 'Hardware')
        self.tabs.addTab(self.cghTab(), 'CGH')
        self.tabs.addTab(self.trapTab(), 'Traps')
        slmtab = QSLMTab(cgh=self.cgh)
        self.tabs.addTab(slmtab, 'SLM')
        self.tabs.currentChanged.connect(slmtab.expose)
        # move Help to end
        self.tabs.tabBar().moveTab(help_index, self.tabs.count() - 1)
        # set current index of other tabs for expose events
        if hwtab.has_content():
            hwtab.index = self.tabs.indexOf(hwtab)
            self.tabs.currentChanged.connect(hwtab.expose)
        slmtab.index = self.tabs.indexOf(slmtab)
        # populate help browser
        self.browser.load('pyfab')
        self.wstage = hwtab.wstage

    def cghTab(self):
        wcgh = QtGui.QWidget()
        layout = tabLayout(wcgh)
        layout.addWidget(self.wcgh)
        return wcgh

    def trapTab(self):
        wtraps = QtGui.QWidget()
        layout = tabLayout(wtraps)
        layout.addWidget(QTrapWidget(self.pattern))
        return wtraps

    def close(self):
        super(QFabWidget, self).close()
        self.pattern.clearTraps()
        self.slm.close()
        self.slm = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QFabWidget()
    sys.exit(app.exec_())
