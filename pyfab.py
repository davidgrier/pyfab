#!/usr/bin/env python

"""pyfab.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
from traps import QTrappingPattern
from QFabGraphicsView import QFabGraphicsView
from QSLM import QSLM
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

    def __init__(self):
        super(pyfab, self).__init__()
        self.init_hardware()
        self.init_ui()
        self.init_configuration()

    def init_hardware(self):
        # video screen
        screen_size = (640, 480)
        self.fabscreen = QFabGraphicsView(
            size=screen_size, gray=True, mirrored=False)
        # DVR
        self.dvr = QFabDVR(source=self.fabscreen.video)
        # spatial light modulator
        self.slm = QSLM()
        # computation pipeline for the trapping pattern
        self.pattern = QTrappingPattern(self.fabscreen)
        self.cgh = CGH(self.slm)
        self.pattern.pipeline = self.cgh

    def init_ui(self):
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.fabscreen)
        wcontrols = QtGui.QWidget()
        controls = QtGui.QVBoxLayout()
        controls.setSpacing(1)
        controls.setSizeConstraint(QtGui.QLayout.SetFixedSize)
        controls.addWidget(self.dvr)
        self.wvideo = QFabVideo(self.fabscreen.video)
        controls.addWidget(self.wvideo)
        controls.addWidget(QFabFilter(self.fabscreen.video))
        controls.addWidget(QCGH(self.cgh))
        wcontrols.setLayout(controls)
        layout.addWidget(wcontrols)
        self.setLayout(layout)
        self.show()
        self.dvr.recording.connect(self.handleRecording)

    def handleRecording(self, recording):
        self.wvideo.enabled = not recording

    def init_configuration(self):
        sz = self.fabscreen.size()
        self.cgh.rc = (sz.width() / 2, sz.height() / 2, 0.)

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
