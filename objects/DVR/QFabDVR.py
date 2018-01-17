from PyQt4 import QtGui, QtCore
from PyQt4.QtCore import Qt
from .fabdvr import fabdvr
from ..clickable import clickable


class QFabDVR(fabdvr, QtGui.QFrame):

    recording = QtCore.pyqtSignal(bool)

    def __init__(self, **kwargs):
        super(QFabDVR, self).__init__(**kwargs)
        self.initUI()

    def initUI(self):
        self.setFrameShape(QtGui.QFrame.Box)
        # Create layout
        layout = QtGui.QGridLayout(self)
        layout.setMargin(1)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(3)
        # Widgets
        iconsize = QtCore.QSize(24, 24)
        title = QtGui.QLabel('Video Recorder')
        self.brecord = QtGui.QPushButton('Record', self)
        self.brecord.clicked.connect(self.handleRecord)
        self.brecord.setIcon(
            self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.brecord.setIconSize(iconsize)
        self.brecord.setToolTip('Start recording video')
        self.bstop = QtGui.QPushButton('Stop', self)
        self.bstop.clicked.connect(self.handleStop)
        self.bstop.setIcon(self.style().standardIcon(
            QtGui.QStyle.SP_MediaStop))
        self.bstop.setIconSize(iconsize)
        self.bstop.setToolTip('Stop recording video')
        self.wframe = self.framecounter_widget()
        wfilelabel = QtGui.QLabel('file name')
        wfilelabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.wfilename = self.filename_widget()
        # Place widgets in layout
        layout.addWidget(title, 1, 1, 1, 3)
        layout.addWidget(self.brecord, 2, 1)
        layout.addWidget(self.bstop, 2, 2)
        layout.addWidget(self.wframe, 2, 3)
        layout.addWidget(wfilelabel, 3, 1)
        layout.addWidget(self.wfilename, 3, 2, 1, 2)
        self.setLayout(layout)

    # customized widgets
    def framecounter_widget(self):
        lcd = QtGui.QLCDNumber(self)
        lcd.setNumDigits(5)
        lcd.setSegmentStyle(QtGui.QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QtGui.QColor(0, 0, 0))
        palette.setColor(palette.Background, QtGui.QColor(255, 255, 255))
        lcd.setPalette(palette)
        lcd.setAutoFillBackground(True)
        return lcd

    def filename_widget(self):
        line = QtGui.QLineEdit()
        line.setText(self.filename)
        line.setReadOnly(True)
        clickable(line).connect(self.getFilename)
        return line

    # core functionality
    def write(self, frame):
        super(QFabDVR, self).write(frame)
        self.wframe.display(self.framenumber)

    def getFilename(self):
        if self.isrecording():
            return
        fn = self.filename
        filename = QtGui.QFileDialog.getSaveFileName(
            self, 'Video File Name', fn, 'Video files (*.avi)')
        if filename:
            self.filename = str(filename)
            self.wfilename.setText(self.filename)

    @QtCore.pyqtSlot()
    def handleRecord(self):
        super(QFabDVR, self).record(1000)
        self.recording.emit(True)

    @QtCore.pyqtSlot()
    def handleStop(self):
        super(QFabDVR, self).stop()
        self.recording.emit(False)


def main():
    import sys
    from QCameraDevice import QCameraDevice
    from QVideoItem import QVideoItem, QVideoWidget

    app = QtGui.QApplication(sys.argv)
    device = QCameraDevice(size=(640, 480), gray=True)
    video = QVideoItem(device)
    widget = QVideoWidget(video, background='w')
    widget.show()
    dvr = QFabDVR(source=video)
    dvr.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
