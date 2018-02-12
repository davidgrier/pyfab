#!/usr/bin/env python

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
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

        self.init_configuration()

    def init_components(self):
        super(QFabWidget, self).init_components()
        # spatial light modulator
        self.slm = QSLM()
        # computation pipeline for the trapping pattern
        # self.computethread = QtCore.QThread()
        # self.cgh.moveToThread(self.computethread)
        # self.computethread.start()
        self.cgh = CGH(slm=self.slm)
        self.wcgh = QCGHPropertyWidget(self.cgh, self.screen)
        # trapping pattern is an interactive overlay
        # that translates user actions into hologram computations
        self.pattern = traps.QTrappingPattern(parent=self.screen)
        self.pattern.sigCompute.connect(self.cgh.setTraps)
        self.cgh.sigComputing.connect(self.pattern.pauseSignals)
        self.cgh.sigHologramReady.connect(self.slm.setData)

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
