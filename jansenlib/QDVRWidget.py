# -*- coding: utf-8 -*-

"""Control panel for DVR functionality."""

from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from video.QVideoPlayer import QVideoPlayer
import cv2
import os
from common.clickable import clickable


class QDVRWidget(QtGui.QFrame):

    recording = QtCore.pyqtSignal(bool)

    def __init__(self,
                 stream=None,
                 filename='~/data/fabdvr.avi',
                 codec='HFYU',
                 **kwargs):
        super(QDVRWidget, self).__init__(**kwargs)

        self._writer = None
        self._player = None
        if cv2.__version__.startswith('2.'):
            self._fourcc = cv2.cv.CV_FOURCC(*codec)
        else:
            self._fourcc = cv2.VideoWriter_fourcc(*codec)
        self._framenumber = 0
        self._nframes = 0

        self.stream = stream
        self.camera = self.stream.source
        self.filename = filename

        self.initUI()

    def initUI(self):
        self.setFrameShape(QtGui.QFrame.Box)
        # Create layout
        layout = QtGui.QGridLayout(self)
        layout.setMargin(1)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(3)
        # Widgets
        self.iconSize = QtCore.QSize(24, 24)
        self.stdIcon = self.style().standardIcon
        title = QtGui.QLabel('Video Recorder')
        self.brecord = self.recordButton()
        self.bstop = self.stopButton()
        self.wframe = self.framecounterWidget()
        wsavelabel = QtGui.QLabel('Save As')
        wsavelabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wsavename = self.saveFilenameWidget()
        self.brewind = self.rewindButton()
        self.bpause = self.pauseButton()
        self.bplay = self.playButton()
        wplaylabel = QtGui.QLabel('Play')
        wplaylabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wplayname = self.playFilenameWidget()
        # Place widgets in layout
        layout.addWidget(title, 1, 1, 1, 3)
        layout.addWidget(self.brecord, 2, 1)
        layout.addWidget(self.bstop, 2, 2)
        layout.addWidget(self.wframe, 2, 3)
        layout.addWidget(wsavelabel, 3, 1)
        layout.addWidget(self.wsavename, 3, 2, 1, 2)
        layout.addWidget(self.brewind, 4, 1)
        layout.addWidget(self.bpause, 4, 2)
        layout.addWidget(self.bplay, 4, 3)
        layout.addWidget(wplaylabel, 5, 1)
        layout.addWidget(self.wplayname, 5, 2, 1, 2)
        self.setLayout(layout)

    def recordButton(self):
        b = QtGui.QPushButton('Record', self)
        b.clicked.connect(self.record)
        b.setIcon(self.stdIcon(QtGui.QStyle.SP_MediaPlay))
        b.setIconSize(self.iconSize)
        b.setToolTip('Start recording video')
        return b

    def stopButton(self):
        b = QtGui.QPushButton('Stop', self)
        b.clicked.connect(self.stop)
        b.setIcon(self.stdIcon(QtGui.QStyle.SP_MediaStop))
        b.setIconSize(self.iconSize)
        b.setToolTip('Stop recording video')
        return b

    def framecounterWidget(self):
        lcd = QtGui.QLCDNumber(self)
        lcd.setNumDigits(5)
        lcd.setSegmentStyle(QtGui.QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QtGui.QColor(0, 0, 0))
        palette.setColor(palette.Background, QtGui.QColor(255, 255, 255))
        lcd.setPalette(palette)
        lcd.setAutoFillBackground(True)
        lcd.setToolTip('Frame counter')
        return lcd

    def saveFilenameWidget(self):
        line = QtGui.QLineEdit()
        line.setText(self.filename)
        line.setReadOnly(True)
        clickable(line).connect(self.getSaveFilename)
        line.setToolTip('Click to change file name')
        return line

    def getSaveFilename(self):
        if self.is_recording():
            return
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Video File Name', self.filename, 'Video files (*.avi)')
        if filename:
            self.filename = str(filename)
            self.wsavename.setText(self.filename)
            self.wplayname.setText(self.filename)

    def rewindButton(self):
        b = QtGui.QPushButton('Rewind', self)
        b.clicked.connect(self.rewind)
        b.setIcon(self.stdIcon(QtGui.QStyle.SP_MediaSkipBackward))
        b.setIconSize(self.iconSize)
        b.setToolTip('Pause video')
        return b

    def playButton(self):
        b = QtGui.QPushButton('Play', self)
        b.clicked.connect(self.play)  # FIXME
        b.setIcon(self.stdIcon(QtGui.QStyle.SP_MediaPlay))
        b.setIconSize(self.iconSize)
        b.setToolTip('Play video')
        return b

    def pauseButton(self):
        b = QtGui.QPushButton('Pause', self)
        b.clicked.connect(self.pause)
        b.setIcon(self.stdIcon(QtGui.QStyle.SP_MediaPause))
        b.setIconSize(self.iconSize)
        b.setToolTip('Pause video')
        return b

    def playFilenameWidget(self):
        line = QtGui.QLineEdit()
        line.setText(self.playname)
        line.setReadOnly(True)
        clickable(line).connect(self.getPlayFilename)
        line.setToolTip('Click to change file name')
        return line

    def getPlayFilename(self):
        if self.is_recording():
            return
        filename = QtGui.QFileDialog.getOpenFileName(
            self, 'Video File Name', self.filename, 'Video files (*.avi)')
        if filename:
            self._playname = str(filename)
            self.wplayname.setText(self.playname)

    # Recording functionality

    @QtCore.pyqtSlot()
    def record(self, nframes=10000):
        if (self.is_recording() or
                self.is_playing() or
                (self.stream is None) or
                (nframes <= 0)):
            return
        self._nframes = nframes
        self._framenumber = 0
        w = self.stream.width()
        h = self.stream.height()
        color = not self.stream.gray
        self._writer = cv2.VideoWriter(self.filename, self._fourcc,
                                       self.stream.fps(), (w, h), color)
        self.stream.sigNewFrame.connect(self.write)
        self.recording.emit(True)

    @QtCore.pyqtSlot()
    def stop(self):
        if self.is_recording():
            self.stream.sigNewFrame.disconnect(self.write)
            self._writer.release()
        if self.is_playing():
            self._player.stop()
            self.stream.source = self.stream.defaultSource()
        self._nframes = 0
        self._writer = None
        self.recording.emit(False)

    def write(self, frame):
        if self.stream.transposed:
            frame = cv2.transpose(frame)
        if self.stream.flipped:
            frame = cv2.flip(frame, 0)
        self._writer.write(frame)
        self._framenumber += 1
        self.wframe.display(self._framenumber)
        if (self._framenumber == self._nframes):
            self.stop()

    # Playback functionality

    @QtCore.pyqtSlot()
    def play(self):
        if self.is_recording():
            return
        self._framenumber = 0
        self._player = QVideoPlayer(self.playname)
        self._player.start()
        self.stream.source = self._player

    @QtCore.pyqtSlot()
    def rewind(self):
        if self.is_playing():
            self._player.rewind()

    @QtCore.pyqtSlot()
    def pause(self):
        if self.is_playing():
            self._player.pause(True)

    # Core capabilities

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not (self.is_recording() or self.is_playing()):
            self._filename = os.path.expanduser(filename)
            self._playname = self._filename

    @property
    def playname(self):
        return self._playname

    @playname.setter
    def playname(self, filename):
        if not (self.is_playing()):
            self._playname = os.path.expanduser(filename)

    def is_recording(self):
        return (self._writer is not None)

    def is_playing(self):
        return (self._player is not None)

    def framenumber(self):
        return self._framenumber
