# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, QThread, QEvent,
                          pyqtSignal, pyqtSlot, pyqtProperty)
from PyQt5.QtWidgets import (QFrame, QFileDialog)
from .QDVRWidget import Ui_QDVRWidget
from .QVideoWriter import QVideoWriter
from .QHDF5Writer import QHDF5Writer
from .QVideoPlayer import QVideoPlayer
from .QHDF5Player import QHDF5Player
import os

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def clickable(widget):
    """Adds a clicked signal to a widget such as QLineEdit that
    ordinarily does not provide notifications of clicks."""

    class Filter(QObject):

        clicked = pyqtSignal()

        def eventFilter(self, obj, event):
            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        return True
            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked


class QDVR(QFrame):

    recording = pyqtSignal(bool)

    def __init__(self,
                 parent=None,
                 source=None,
                 screen=None,
                 filename='~/data/fabdvr.avi'):
        super(QDVR, self).__init__(parent)

        self._writer = None
        self._player = None
        self._framenumber = 0
        self._nframes = 0

        self.ui = Ui_QDVRWidget()
        self.ui.setupUi(self)
        self.connectSignals()

        self.source = source
        self.screen = screen
        self.filename = filename

    def connectSignals(self):
        clickable(self.ui.playEdit).connect(self.getPlayFilename)
        clickable(self.ui.saveEdit).connect(self.getSaveFilename)
        self.ui.recordButton.clicked.connect(self.record)
        self.ui.stopButton.clicked.connect(self.stop)
        self.ui.rewindButton.clicked.connect(self.rewind)
        self.ui.pauseButton.clicked.connect(self.pause)
        self.ui.playButton.clicked.connect(self.play)

    def is_recording(self):
        return (self._writer is not None)

    def is_playing(self):
        return (self._player is not None)

    # =====
    # Slots
    #

    @pyqtSlot()
    def getPlayFilename(self):
        if self.is_recording():
            return
        filename, filter = QFileDialog.getOpenFileName(
            self, 'Video File Name', self.filename,
            'Video files (*.avi);;HDF5 files (*.h5)')
        if filename:
            self.playname = str(filename)

    @pyqtSlot()
    def getSaveFilename(self):
        if self.is_recording():
            return
        filename, filter = QFileDialog.getSaveFileName(
            self, 'Video File Name', self.filename,
            'Video files (*.avi);;HDF5 files (*.h5)')
        if filename:
            self.filename = str(filename)

    # Record functionality

    @pyqtSlot()
    def record(self, nframes=10000):
        if (self.is_recording() or self.is_playing() or (nframes <= 0)):
            return
        logger.debug('Starting Recording')
        if os.path.splitext(self.filename) == 'avi':
            self._writer = QVideoWriter(self.filename, self.source.shape,
                                        nframes=nframes)
        else:
            self._writer = QHDF5Writer(self.filename, nframes=nframes)
        self._writer.sigFrameNumber.connect(self.setFrameNumber)
        self._writer.sigFinished.connect(self.stop)
        self._thread = QThread()
        self._thread.finished.connect(self._writer.close)
        self.source.sigNewFrame.connect(self._writer.write)
        self._writer.moveToThread(self._thread)
        self._thread.start()
        self.recording.emit(True)

    @pyqtSlot()
    def stop(self):
        if self.is_recording():
            logger.debug('Stopping Recording')
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._writer = None
        if self.is_playing():
            logger.debug('Stopping Playing')
            self._player.stop()
            self._player = None
            self.screen.source = None  # use default source
        self.framenumber = 0
        self._nframes = 0
        self.recording.emit(False)

    @pyqtSlot(int)
    def setFrameNumber(self, framenumber):
        self.framenumber = framenumber

    # Playback functionality

    @pyqtSlot()
    def play(self):
        if self.is_recording():
            return
        if self.is_playing():
            self._player.pause(False)
            return
        logger.debug('Starting Playback')
        self.framenumber = 0
        if os.path.splitext(self.playname) == 'avi':
            self._player = QVideoPlayer(self.playname)
        else:
            self._player = QHDF5Player(self.playname)
        if self._player.isOpened():
            self._player.sigNewFrame.connect(self.stepFrameNumber)
            self._player.start()
            self.screen.source = self._player
        else:
            self._player = None

    @pyqtSlot()
    def rewind(self):
        if self.is_playing():
            self._player.rewind()
            self.framenumber = 0

    @pyqtSlot()
    def pause(self):
        if self.is_playing():
            state = self._player.isPaused()
            self._player.pause(not state)

    @pyqtSlot()
    def stepFrameNumber(self):
        self.framenumber += 1

    # ==========
    # Properties
    #

    @pyqtProperty(QObject)
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source
        self.ui.recordButton.setEnabled(source is not None)

    @pyqtProperty(QObject)
    def screen(self):
        return self._screen

    @screen.setter
    def screen(self, screen):
        self._screen = screen
        self.ui.playButton.setEnabled(screen is not None)

    @pyqtProperty(str)
    def filename(self):
        return str(self.ui.saveEdit.text())

    @filename.setter
    def filename(self, filename):
        if not (self.is_recording() or self.is_playing()):
            self.ui.saveEdit.setText(os.path.expanduser(filename))
            self.playname = self.filename

    @pyqtProperty(str)
    def playname(self):
        return str(self.ui.playEdit.text())

    @playname.setter
    def playname(self, filename):
        if not (self.is_playing()):
            self.ui.playEdit.setText(os.path.expanduser(filename))

    @pyqtProperty(int)
    def framenumber(self):
        return self._framenumber

    @framenumber.setter
    def framenumber(self, number):
        self._framenumber = number
        self.ui.frameNumber.display(self._framenumber)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    wid = QDVR()
    wid.show()
    sys.exit(app.exec_())
