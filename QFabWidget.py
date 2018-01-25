#!/usr/bin/env python

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui, QtCore
from QJansenWidget import QJansenWidget
import traps
import objects
import sys
import logging

def tabLayout():
    layout = QtGui.QVBoxLayout()
    layout.setAlignment(QtCore.Qt.AlignTop)
    layout.setSpacing(1)
    return layout
    
class hardwareTab(QtGui.QWidget):

    def __init__(self, parent):
        super(hardwareTab, self).__init__(parent=parent)
        self.title = 'Hardware'
        self.index = -1

        layout = tabLayout()
        try:
            self.wstage = objects.QProscan()
            layout.addWidget(self.wstage)            
        except ValueError as ex:
            self.wstage = None
            logging.warning('Could not install stage: %s', ex)
        try:
            self.wlaser = objects.QIPGLaser()
            layout.addWidget(self.wlaser)
        except ValueError as ex:
            self.wlaser = None
            logging.warning('Could not install laser: %s', ex)
        self.setLayout(layout)
        self.parent().tabs.currentChanged.connect(self.expose)

    def expose(self, index):
        if index == self.index:
            if self.wstage is not None: self.wstage.start()
            if self.wlaser is not None: self.wlaser.start()
        else:
            if self.wstage is not None: self.wstage.stop()
            if self.wlaser is not None: self.wstage.stop()

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
