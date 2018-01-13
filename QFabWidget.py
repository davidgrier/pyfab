#!/usr/bin/env python

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui, QtCore
from QJansenWidget import QJansenWidget
import traps
import objects
import sys


class QFabWidget(QJansenWidget):

    def __init__(self, size=(640, 480)):
        super(QFabWidget, self).__init__(size=size)
        self.init_configuration()

    def init_hardware(self, size):
        super(QFabWidget, self).init_hardware(size)
        # stage
        self.wstage = objects.QProscan()
        # spatial light modulator
        self.slm = objects.QSLM()
        # computation pipeline for the trapping pattern
        try:
            self.cgh = objects.cudaCGH(slm=self.slm)
        except (NameError, AttributeError):
            print('could not load cudaCGH')
            self.cgh = objects.CGH(slm=self.slm)
        # self.computethread = QtCore.QThread()
        # self.cgh.moveToThread(self.computethread)
        # self.computethread.start()
        self.wcgh = objects.QCGH(self.cgh)
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
        layout.addWidget(self.wstage)
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
