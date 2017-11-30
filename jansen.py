#!/usr/bin/env python

"""jansen.py: GUI for holographic video microscopy."""

from pyqtgraph.Qt import QtGui
from traps import QTrappingPattern
from QFabGraphicsView import QFabGraphicsView
from QFabDVR import QFabDVR
from QFabVideo import QFabVideo
from QFabFilter import QFabFilter
import sys


class jansen(QtGui.QWidget):

    def __init__(self):
        super(jansen, self).__init__()
        self.init_hardware()
        self.init_ui()

    def init_hardware(self):
        # video screen
        screen_size = (640, 480)
        self.fabscreen = QFabGraphicsView(
            size=screen_size, gray=True, mirrored=False)
        # DVR
        self.dvr = QFabDVR(source=self.fabscreen.video)

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
        wcontrols.setLayout(controls)
        layout.addWidget(wcontrols)
        self.setLayout(layout)
        self.show()
        self.dvr.recording.connect(self.handleRecording)

    def handleRecording(self, recording):
        self.wvideo.enabled = not recording


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = jansen()
    sys.exit(app.exec_())
