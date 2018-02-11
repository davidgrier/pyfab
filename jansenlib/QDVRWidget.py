from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
import cv2
import os
from common.clickable import clickable


class QDVRWidget(QtGui.QFrame):

    recording = QtCore.pyqtSignal(bool)

    def __init__(self,
                 source=None,
                 filename='~/data/fabdvr.avi',
                 codec='HFYU',
                 **kwargs):
        super(QDVRWidget, self).__init__(**kwargs)

        self.source = source
        self._writer = None
        self._framenumber = 0
        self._nframes = 0
        if cv2.__version__.startswith('2.'):
            self._fourcc = cv2.cv.CV_FOURCC(*codec)
        else:
            self._fourcc = cv2.VideoWriter_fourcc(*codec)
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
        wfilelabel = QtGui.QLabel('file name')
        wfilelabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wfilename = self.filenameWidget()
        # Place widgets in layout
        layout.addWidget(title, 1, 1, 1, 3)
        layout.addWidget(self.brecord, 2, 1)
        layout.addWidget(self.bstop, 2, 2)
        layout.addWidget(self.wframe, 2, 3)
        layout.addWidget(wfilelabel, 3, 1)
        layout.addWidget(self.wfilename, 3, 2, 1, 2)
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
        return lcd

    def filenameWidget(self):
        line = QtGui.QLineEdit()
        line.setText(self.filename)
        line.setReadOnly(True)
        clickable(line).connect(self.getFilename)
        return line

    def getFilename(self):
        if self.is_recording():
            return
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Video File Name', self.filename, 'Video files (*.avi)')
        if filename:
            self._filename = str(filename)
            self.wfilename.setText(self._filename)

    @QtCore.pyqtSlot()
    def record(self, nframes=10000):
        if (self.is_recording() or (self.source is None) or
                (nframes <= 0)):
            return
        self._nframes = nframes
        self._framenumber = 0
        w = self.source.device.width
        h = self.source.device.height
        color = not self.source.gray
        self._writer = cv2.VideoWriter(self.filename, self._fourcc,
                                       self.source.fps, (w, h), color)
        self.source.sigNewFrame.connect(self.write)
        self.recording.emit(True)

    @QtCore.pyqtSlot()
    def stop(self):
        if self.is_recording():
            self.source.sigNewFrame.disconnect(self.write)
            self._writer.release()
        self._nframes = 0
        self._writer = None
        self.recording.emit(False)

    def write(self, frame):
        if self.source.transposed:
            frame = cv2.transpose(frame)
        if self.source.flipped:
            frame = cv2.flip(frame, 0)
        self._writer.write(frame)
        self._framenumber += 1
        self.wframe.display(self._framenumber)
        if (self._framenumber == self._nframes):
            self.stop()

    # Core capabilities
    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not self.is_recording():
            self._filename = os.path.expanduser(filename)

    def is_recording(self):
        return (self._writer is not None)

    def framenumber(self):
        return self._framenumber
