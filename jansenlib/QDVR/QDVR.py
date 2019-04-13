# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSignal, pyqtSlot, QEvent)
from PyQt5.QtWidgets import (QFrame, QFileDialog)
from .QDVRWidget import Ui_QDVRWidget
from jansenlib.video.QVideoPlayer import QVideoPlayer
import cv2
import numpy as np
import os
import platform

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


class Writer(QObject):
    '''NOTE: Move writing to separate thread'''

    sigFrameNumber = pyqtSignal(int)
    sigFinished = pyqtSignal()

    def __init__(self, parent, nframes=10000):
        super(Writer, self).__init__(parent)
        self.shape = self.parent().source.shape
        color = (len(self.shape) == 3)
        h, w = self.shape[0:2]
        fps = 24
        msg = 'Recording: {}x{}, color: {}, fps: {}'
        logger.info(msg.format(w, h, color, fps))
        self.writer = cv2.VideoWriter(self.parent().filename,
                                      self.parent()._fourcc,
                                      fps, (w, h), color)
        self.framenumber = 0
        self.target = nframes
        self.sigFrameNumber.emit(self.framenumber)

    @pyqtSlot(np.ndarray)
    def write(self, frame):
        self.writer.write(frame)
        self.framenumber += 1
        self.sigFramenumber.emit(self.framenumber)
        if (self.framenumber >= self.target):
            self.sigFinished.emit()

    def stop(self):
        self.writer.release()


class QDVR(QFrame):

    recording = pyqtSignal(bool)

    def __init__(self,
                 parent=None,
                 source=None,
                 screen=None,
                 filename='~/data/fabdvr.avi',
                 codec=None):
        super(QDVR, self).__init__(parent)

        self._writer = None
        self._player = None
        if codec is None:
            if platform.system() == 'Linux':
                codec = 'HFYU'
            else:
                codec = 'X264'
        if cv2.__version__.startswith('2.'):
            self._fourcc = cv2.cv.CV_FOURCC(*codec)
        else:
            self._fourcc = cv2.VideoWriter_fourcc(*codec)
        self._framenumber = 0
        self._nframes = 0

        self.ui = Ui_QDVRWidget()
        self.ui.setupUi(self)
        self.configureUi()
        self.connectSignals()

        self.source = source
        self.screen = screen
        self.filename = filename

    def configureUi(self):
        icon = self.style().standardIcon
        # self.ui.recordButton.setIcon(icon(QStyle.SP_MediaPlay))
        # self.ui.stopButton.setIcon(icon(QStyle.SP_MediaStop))
        # self.ui.rewindButton.setIcon(icon(QStyle.SP_MediaSkipBackward))
        # self.ui.playButton.setIcon(icon(QStyle.SP_MediaPlay))
        # self.ui.pauseButton.setIcon(icon(QStyle.SP_MediaPause))

    def connectSignals(self):
        clickable(self.ui.playEdit).connect(self.getPlayFilename)
        clickable(self.ui.saveEdit).connect(self.getSaveFilename)
        self.ui.recordButton.clicked.connect(self.recordThread)
        self.ui.stopButton.clicked.connect(self.stopThread)
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
        filename, _filter = QFileDialog.getOpenFileName(
            self, 'Video File Name', self.filename, 'Video files (*.avi)')
        if filename:
            self.playname = str(filename)

    @pyqtSlot()
    def getSaveFilename(self):
        if self.is_recording():
            return
        filename, _filter = QFileDialog.getSaveFileName(
            self, 'Video File Name', self.filename, 'Video files (*.avi)')
        if filename:
            self.filename = str(filename)

    # Record functionality

    @pyqtSlot()
    def recordThread(self, nframes=10000):
        if (self.is_recording() or self.is_playing() or (nframes <= 0)):
            return
        logger.debug('Starting Threaded Recording')
        self._writer = Writer(self)
        self.source.sigNewFrame.connect(self._writer.write)
        self._writer.sigFrameNumber.connect(self.setFrameNumber)
        self._writer.sigFinished.connect(self.stopThread)
        self._thread = QThread()
        self._writer.moveToThread(self._thread)
        self._thread.start()
        self.recording.emit(True)

    @pyqtSlot()
    def stopThread(self):
        if self.is_recording():
            logger.debug('Stopping Threaded Recording')
            self.source.sigNewFrame.disconnect(self._writer.write)
            self._writer.sigFrameNumber.disconnect(self.setFrameNumber)
            self._writer.sigFinished.disconnect(self.stopThread)
            self._thread.stop()
            del self._writer
            self._writer = None
        if self.is_playing():
            logger.debug('Stopping Playing')
            self._player.stop()
            self._player = None
            self.screen.source = self.screen.camera
        self.framenumber = 0
        self._nframes = 0
        self.recording.emit(False)

    @pyqtSlot(int)
    def setFrameNumber(self, framenumber):
        self.framenumber = framenumber

    @pyqtSlot(bool)
    def record(self, state, nframes=10000):
        if (self.is_recording() or self.is_playing() or
                (nframes <= 0)):
            return
        logger.debug('Starting Recording')
        self._nframes = nframes
        self.framenumber = 0
        fps = 24.  # self.screen.fps()
        self._shape = self.source.shape
        color = (len(self._shape) == 3)
        h, w = self._shape[0:2]
        msg = 'Recording: {}x{}, color: {}, fps: {}'
        logger.info(msg.format(w, h, color, fps))
        self._writer = cv2.VideoWriter(self.filename, self._fourcc,
                                       fps, (w, h), color)
        self.source.sigNewFrame.connect(self.write)
        self.recording.emit(True)

    @pyqtSlot()
    def stop(self):
        if self.is_recording():
            logger.debug('Stopping Recording')
            self.source.sigNewFrame.disconnect(self.write)
            self._writer.release()
            self._writer = None
        if self.is_playing():
            logger.debug('Stopping Playing')
            self._player.stop()
            self._player = None
            self.screen.source = self.screen.camera
        self.framenumber = 0
        self._nframes = 0
        self.recording.emit(False)

    @pyqtSlot(np.ndarray)
    def write(self, frame):
        if not self.is_recording():
            logger.debug('Tried to write past end of video')
            return
        if frame.shape != self._shape:
            msg = 'Frame is wrong shape: {}, expecting: {}'
            logger.warn(msg.format(frame.shape, self._shape))
            self.stop()
            return
        logger.debug('Frame: {}'.format(frame.shape))
        self._writer.write(frame)
        self.framenumber += 1
        if self.framenumber >= self._nframes:
            self.stop()

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
        self._player = QVideoPlayer(self, self.playname)
        self._player.sigNewFrame.connect(self.stepFrameNumber)
        self._player.start()
        self.screen.source = self._player

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

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        self._source = source
        self.ui.recordButton.setEnabled(source is not None)

    @property
    def screen(self):
        return self._screen

    @screen.setter
    def screen(self, screen):
        self._screen = screen
        self.ui.playButton.setEnabled(screen is not None)

    @property
    def filename(self):
        return str(self.ui.saveEdit.text())

    @filename.setter
    def filename(self, filename):
        if not (self.is_recording() or self.is_playing()):
            self.ui.saveEdit.setText(os.path.expanduser(filename))
            self.playname = self.filename

    @property
    def playname(self):
        return str(self.ui.playEdit.text())

    @playname.setter
    def playname(self, filename):
        if not (self.is_playing()):
            self.ui.playEdit.setText(os.path.expanduser(filename))

    @property
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
