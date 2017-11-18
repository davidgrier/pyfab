from PyQt4 import QtGui
from fabdvr import fabdvr


class QFabDVR(fabdvr, QtGui.QWidget):

    def __init__(self, **kwargs):
        super(QFabDVR, self).__init__(**kwargs)
        self.initUI()

    def initUI(self):
        layout = QtGui.QGridLayout(self)
        self.brecord = QtGui.QPushButton('Record', self)
        self.bstop = QtGui.QPushButton('Stop', self)
        self.wframe = QtGui.QLCDNumber(self)
        self.wframe.setNumDigits(5)
        self.wfilename = QtGui.QLabel()
        self.wfilename.setText(self.filename)
        layout.addWidget(self.brecord, 1, 1)
        layout.addWidget(self.bstop, 1, 2)
        layout.addWidget(self.wframe, 1, 3)
        layout.addWidget(self.wfilename, 2, 1, 1, 3)
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
