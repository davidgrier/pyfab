#!/usr/bin/env python

"""pyfab.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
from traps import QTrappingPattern, QTrapWidget
from QFabGraphicsView import QFabGraphicsView
from QSLM import QSLM
try:
    from cudaCGH import cudaCGH
except ImportError:
    from CGH import CGH
from QCGH import QCGH
from QFabDVR import QFabDVR
from QFabVideo import QFabVideo
from QFabFilter import QFabFilter
import sys
import io
import datetime
import os


class pyfab(QtGui.QWidget):

    def __init__(self, size=(640, 480)):
        super(pyfab, self).__init__()
        self.init_hardware(size)
        self.init_ui()
        self.init_configuration()

    def init_hardware(self, size):
        # video screen
        self.fabscreen = QFabGraphicsView(size=size, gray=True)
        self.video = QFabVideo(self.fabscreen.video)
        self.filters = QFabFilter(self.fabscreen.video)
        # DVR
        self.dvr = QFabDVR(source=self.fabscreen.video)
        self.dvr.recording.connect(self.handleRecording)
        # spatial light modulator
        self.slm = QSLM()
        # computation pipeline for the trapping pattern
        try:
            self.cgh = cudaCGH(self.slm)
        except NameError:
            self.cgh = CGH(self.slm)
        self.pattern = QTrappingPattern(self.fabscreen)
        self.pattern.pipeline = self.cgh

    def init_ui(self):
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.fabscreen)
        tabs = QtGui.QTabWidget()
        tabs.addTab(self.controlTab(), 'Controls')
        tabs.addTab(self.trapTab(), 'Traps')
        layout.addWidget(tabs)
        self.setLayout(layout)
        self.show()
        tabs.setFixedSize(tabs.size())

    def controlTab(self):
        wcontrols = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setSpacing(1)
        layout.addWidget(self.dvr)
        layout.addWidget(self.video)
        layout.addWidget(self.filters)
        layout.addWidget(QCGH(self.cgh))
        wcontrols.setLayout(layout)
        return wcontrols

    def trapTab(self):
        wtraps = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setSpacing(1)
        layout.addWidget(QTrapWidget(self.pattern))
        wtraps.setLayout(layout)
        return wtraps

    def handleRecording(self, recording):
        self.video.enabled = not recording

    def init_configuration(self):
        sz = self.fabscreen.video.device.size
        self.cgh.rc = (sz.width() / 2, sz.height() / 2, 0.)
        sz = self.slm.size()
        self.cgh.rs = (sz.width() / 2, sz.height() / 2)

    def save_configuration(self):
        scgh = self.cgh.serialize()
        tn = datetime.datetime.now()
        fn = '~/.pyfab/pyfab_{:%Y%b%d_%H:%M:%S}.json'.format(tn)
        fn = os.path.expanduser(fn)
        with io.open(fn, 'w', encoding='utf8') as configfile:
            configfile.write(unicode(scgh))

    def closeEvent(self, event):
        self.save_configuration()
        self.slm.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = pyfab()
    sys.exit(app.exec_())
