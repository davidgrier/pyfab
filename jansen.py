#!/usr/bin/env python

"""jansen.py: GUI for holographic video microscopy."""

from pyqtgraph.Qt import QtGui, QtCore
import objects
import sys


class jansen(QtGui.QWidget):

    def __init__(self, size=(640, 480)):
        super(jansen, self).__init__()
        self.init_hardware(size)
        self.init_ui()

    def init_hardware(self, size):
        # video screen
        self.fabscreen = objects.QFabScreen(size=size, gray=True)
        self.video = objects.QFabVideo(self.fabscreen.video)
        self.filters = objects.QFabFilter(self.fabscreen.video)
        # DVR
        self.dvr = objects.QFabDVR(source=self.fabscreen.video)
        self.dvr.recording.connect(self.handleRecording)

    def init_ui(self):
        layout = QtGui.QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(self.fabscreen)
        self.tabs = QtGui.QTabWidget()
        self.tabs.addTab(self.videoTab(), 'Video')
        layout.addWidget(self.tabs)
        layout.setAlignment(self.tabs, QtCore.Qt.AlignTop)
        self.setLayout(layout)
        self.show()
        self.tabs.setFixedWidth(self.tabs.width())

    def videoTab(self):
        wvideo = QtGui.QWidget()
        layout = QtGui.QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignTop)
        layout.setSpacing(1)
        layout.addWidget(self.dvr)
        layout.addWidget(self.video)
        layout.addWidget(self.filters)
        wvideo.setLayout(layout)
        return wvideo

    def handleRecording(self, recording):
        self.video.enabled = not recording


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    instrument = jansen()
    sys.exit(app.exec_())
