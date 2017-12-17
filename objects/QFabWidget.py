#!/usr/bin/env python

"""QFabWidget.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui, QtCore
from jansen import jansen
import traps
import objects
import sys
import io
import datetime
import os
import json


class QFabWidget(jansen):

    def __init__(self, size=(640, 480)):
        super(QFabWidget, self).__init__(size=size)
        self.init_configuration()

    def init_hardware(self, size):
        super(QFabWidget, self).init_hardware(size)
        # spatial light modulator
        self.slm = objects.QSLM()
        # computation pipeline for the trapping pattern
        try:
            self.cgh = objects.cudaCGH(self.slm)
        except (NameError, AttributeError):
            self.cgh = objects.CGH(self.slm)
        self.wcgh = objects.QCGH(self.cgh)
        self.pattern = traps.QTrappingPattern(self.fabscreen)
        self.pattern.pipeline = self.cgh

    def init_ui(self):
        super(QFabWidget, self).init_ui()
        self.tabs.addTab(self.cghTab(), 'CGH')
        self.tabs.addTab(self.trapTab(), 'Traps')

    def cghTab(self):
        wcgh = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(1)
        layout.addWidget(self.wcgh)
        wcgh.setLayout(layout)
        return wcgh

    def trapTab(self):
        wtraps = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(1)
        layout.addWidget(traps.QTrapWidget(self.pattern))
        wtraps.setLayout(layout)
        return wtraps

    def init_configuration(self):
        sz = self.fabscreen.video.device.size
        fn = '~/.pyfab/pyfab.json'
        fn = os.path.expanduser(fn)
        try:
            values = json.load(io.open(fn))
            self.wcgh.calibration = values
        except IOError:
            self.wcgh.xc = sz.width() / 2
            self.wcgh.yc = sz.height() / 2
            self.wcgh.zc = 0.
            sz = self.slm.size()
            self.wcgh.xs = sz.width() / 2
            self.wcgh.ys = sz.height() / 2

    def save_configuration(self):
        scgh = self.wcgh.serialize()
        tn = datetime.datetime.now()
        fn = '~/.pyfab/pyfab_{:%Y%b%d_%H:%M:%S}.json'.format(tn)
        fn = os.path.expanduser(fn)
        with io.open(fn, 'w', encoding='utf8') as configfile:
            configfile.write(unicode(scgh))
        fn = '~/.pyfab/pyfab.json'
        fn = os.path.expanduser(fn)
        with io.open(fn, 'w', encoding='utf8') as configfile:
            configfile.write(unicode(scgh))

    def query_save_configuration(self):
        query = 'Save current configuration?'
        reply = QtGui.QMessageBox.question(self, 'Confirmation',
                                           query,
                                           QtGui.QMessageBox.Yes,
                                           QtGui.QMessageBox.No)
        if reply == QtGui.QMessageBox.Yes:
            self.save_configuration()
        else:
            pass

    def closeEvent(self, event):
        self.query_save_configuration()
        self.slm.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    QFabWidget()
    sys.exit(app.exec_())
