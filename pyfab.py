#!/usr/bin/env python

"""pyfab.py: GUI for holographic optical trapping."""

from pyqtgraph.Qt import QtGui
from traps import QTrappingPattern
from QFabGraphicsView import QFabGraphicsView
from QSLM import QSLM
from CGH import CGH
from QCGH import QCGH
from QFabDVR import QFabDVR
from QFabCamera import QFabCamera
import sys


class pyfab(QtGui.QWidget):

    def __init__(self):
        super(pyfab, self).__init__()
        self.init_hardware()
        self.init_ui()
        self.init_calibration()

    def init_hardware(self):
        # video screen
        screen_size = (640, 480)
        self.fabscreen = QFabGraphicsView(
            size=screen_size, gray=True, mirrored=False)
        # DVR
        self.dvr = QFabDVR(camera=self.fabscreen.camera)
        # spatial light modulator
        self.slm = QSLM(fake=True)
        # computation pipeline for the trapping pattern
        self.pattern = QTrappingPattern(self.fabscreen)
        self.cgh = CGH(self.slm)
        self.pattern.pipeline = self.cgh

    def init_ui(self):
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.fabscreen)
        controls = QtGui.QVBoxLayout()
        controls.setSpacing(1)
        controls.setSizeConstraint(QtGui.QLayout.SetFixedSize)
        controls.addWidget(self.dvr)
        self.wcamera = QFabCamera(self.fabscreen.camera.device)
        controls.addWidget(self.wcamera)
        controls.addWidget(QCGH(self.cgh))
        controls.addItem(QtGui.QSpacerItem(
            20, 40, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding))
        layout.addItem(controls)
        self.setLayout(layout)
        self.show()
        self.dvr.recording.connect(self.handleRecording)

    def handleRecording(self, recording):
        if recording:
            self.wcamera.enabled = False
        else:
            self.wcamera.enabled = True

    def init_calibration(self):
        sz = self.fabscreen.size()
        self.cgh.rc = (sz.width() / 2, sz.height() / 2)

    def closeEvent(self, event):
        self.slm.close()


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = pyfab()
    sys.exit(app.exec_())
