#!/usr/bin/env python

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui, QtCore
from jansenlib.QJansenWidget import QJansenWidget
from QHardwareTab import QHardwareTab
from common.tabLayout import tabLayout
import traps
from QSLM import QSLM
from CGH import CGH, QCGHPropertyWidget
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
        self.wcgh = QCGHPropertyWidget(self.cgh, self.screen)
        # trapping pattern is an interactive overlay
        # that translates user actions into hologram computations
        self.pattern = traps.QTrappingPattern(parent=self.screen)
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
        hw_tab = QHardwareTab()
        self.tabs.addTab(hw_tab, 'Hardware')
        self.tabs.addTab(self.cghTab(), 'CGH')
        self.tabs.addTab(self.trapTab(), 'Traps')
        # move Help to end
        self.tabs.tabBar().moveTab(help_index, self.tabs.count() - 1)
        # set current index of hardware tab for expose events
        hw_tab.index = self.tabs.indexOf(hw_tab)
        self.tabs.currentChanged.connect(hw_tab.expose)
        # populate help browser
        self.browser.load('pyfab')
        self.wstage = hw_tab.wstage

    def cghTab(self):
        wcgh = QtGui.QWidget()
        layout = tabLayout(wcgh)
        layout.addWidget(self.wcgh)
        return wcgh

    def trapTab(self):
        wtraps = QtGui.QWidget()
        layout = tabLayout(wtraps)
        layout.addWidget(traps.QTrapWidget(self.pattern))
        return wtraps

    def close(self):
        self.pattern.clearTraps()
        self.slm.close()
        self.slm = None
        self.thread.quit()
        self.thread.wait()
        self.thread = None

    def closeEvent(self, event):
        self.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QFabWidget()
    sys.exit(app.exec_())
