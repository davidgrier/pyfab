from PyQt4 import QtGui, QtCore
from .pyproscan import pyproscan
import atexit


class QProscan(QtGui.QFrame):

    def __init__(self):
        super(QProscan, self).__init__()
        self.instrument = pyproscan()
        self.initUI()
        atexit.register(self.shutdown)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.setInterval(100)
        self._timer.start()

    def shutdown(self):
        self._timer.stop()
        self.instrument.close()

    def initUI(self):
        self.setFrameShape(QtGui.QFrame.Box)
        self.wx = self.counter_widget()
        self.wy = self.counter_widget()
        self.wz = self.counter_widget()
        layout = QtGui.QGridLayout()
        layout.setMargin(1)
        layout.setHorizontalSpacing(1)
        layout.setVerticalSpacing(1)
        layout.addWidget(QtGui.QLabel('Stage'), 1, 1, 1, 3)
        layout.addWidget(QtGui.QLabel('x'), 2, 1)
        layout.addWidget(QtGui.QLabel('y'), 2, 2)
        layout.addWidget(QtGui.QLabel('z'), 2, 3)
        layout.addWidget(self.wx, 3, 1)
        layout.addWidget(self.wy, 3, 2)
        layout.addWidget(self.wz, 3, 3)
        self.setLayout(layout)

    def counter_widget(self):
        lcd = QtGui.QLCDNumber(self)
        lcd.setNumDigits(9)
        lcd.setSegmentStyle(QtGui.QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QtGui.QColor(0, 0, 0))
        palette.setColor(palette.Background, QtGui.QColor(255, 255, 224))
        lcd.setPalette(palette)
        lcd.setAutoFillBackground(True)
        return lcd

    def update(self):
        position = self.instrument.position()
        self.wx.display(position[0])
        self.wy.display(position[1])
        self.wz.display(position[2])


def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    stage = QProscan()
    stage.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
