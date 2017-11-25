from PyQt4 import QtGui
from fabdvr import fabdvr


class QFabDVR(fabdvr, QtGui.QFrame):

    def __init__(self, **kwargs):
        super(QFabDVR, self).__init__(**kwargs)
        self.initUI()

    def initUI(self):
        self.setFrameShape(QtGui.QFrame.Box)
        layout = QtGui.QGridLayout(self)
        title = QtGui.QLabel('Video Recorder')
        self.brecord = QtGui.QPushButton('Record', self)
        self.bstop = QtGui.QPushButton('Stop', self)
        self.wframe = QtGui.QLCDNumber(self)
        self.wframe.setNumDigits(5)
        self.wfilename = QtGui.QLabel()
        self.wfilename.setText(self.filename)
        layout.addWidget(title, 1, 1, 1, 3)
        layout.addWidget(self.brecord, 2, 1)
        layout.addWidget(self.bstop, 2, 2)
        layout.addWidget(self.wframe, 2, 3)
        layout.addWidget(self.wfilename, 3, 1, 1, 3)
        self.setLayout(layout)
        self.brecord.clicked.connect(self.handleRecord)
        self.bstop.clicked.connect(self.handleStop)

    def write(self, frame):
        super(QFabDVR, self).write(frame)
        self.wframe.display(self.framenumber)

    def handleRecord(self):
        super(QFabDVR, self).record(1000)

    def handleStop(self):
        super(QFabDVR, self).stop()


def main():
    import sys
    from QCameraItem import QCameraDevice, QCameraItem, QCameraWidget

    app = QtGui.QApplication(sys.argv)
    device = QCameraDevice(size=(640, 480), gray=True)
    camera = QCameraItem(device)
    widget = QCameraWidget(camera, background='w')
    widget.show()
    dvr = QFabDVR(camera=camera)
    dvr.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
