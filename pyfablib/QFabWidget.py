#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui, QtCore
from jansenlib.QJansenWidget import QJansenWidget
from .QHardwareTab import QHardwareTab
from .QSLMTab import QSLMTab
from common.tabLayout import tabLayout
from .traps import QTrappingPattern, QTrapWidget
from .QSLM import QSLM
from .CGH import CGH, QCGHPropertyWidget
import sys


class QFabWidget(QJansenWidget):

    def __init__(self, parent=None):
        super(QFabWidget, self).__init__(parent)

    def init_components(self):
        super(QFabWidget, self).init_components()
        # spatial light modulator
        self.slm = QSLM()

        # computation pipeline for the trapping pattern
        self.cgh = CGH(slm=self.slm)
        self.cgh.sigHologramReady.connect(self.slm.setData)
        self.wcgh = QCGHPropertyWidget(self)

        # move CGH pipeline to a separate thread to readuce latency
        self.thread = QtCore.QThread()
        self.cgh.moveToThread(self.thread)
        self.thread.started.connect(self.cgh.start)
        self.thread.finished.connect(self.cgh.stop)
        self.thread.start()

        # trapping pattern is an interactive overlay
        # that translates user actions into hologram computations
        self.pattern = QTrappingPattern(parent=self)
        self.screen.addOverlay(self.pattern)
        self.screen.sigMousePress.connect(self.pattern.mousePress)
        self.screen.sigMouseRelease.connect(self.pattern.mouseRelease)
        self.screen.sigMouseMove.connect(self.pattern.mouseMove)
        self.screen.sigMouseWheel.connect(self.pattern.mouseWheel)
        self.pattern.sigCompute.connect(self.cgh.setTraps)
        self.cgh.sigComputing.connect(self.screen.pauseSignals)

    def init_ui(self):
        super(QFabWidget, self).init_ui()

        # Insert new tabs before Help tab
        hwtab = QHardwareTab()
        index = self.tabs.insertTab(self.tabs.count()-1, hwtab, 'Hardware')
        self.tabs.setTabToolTip(index, 'Hardware')
        hwtab.index = index
        self.tabs.currentChanged.connect(hwtab.expose)
        self.tabs.setTabEnabled(index, hwtab.has_content())

        cghtab = self.cghTab()
        index = self.tabs.insertTab(index+1, cghtab, 'CGH')
        self.tabs.setTabToolTip(index, 'CGH')

        traptab = self.trapTab()
        index = self.tabs.insertTab(index+1, traptab, 'Traps')
        self.tabs.setTabToolTip(index, 'Traps')

        slmtab = QSLMTab(cgh=self.cgh)
        index = self.tabs.insertTab(index+1, slmtab, 'SLM')
        self.tabs.setTabToolTip(index, 'SLM')
        slmtab.index = index
        self.tabs.currentChanged.connect(slmtab.expose)

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
