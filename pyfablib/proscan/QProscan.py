# -*- coding: utf-8 -*-

"""Control panel for Prior Proscan stage controller."""

from PyQt5.QtWidgets import (QFrame, QLabel, QPushButton,
                             QLCDNumber, QGridLayout)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QColor
from .pyproscan import pyproscan
import atexit


class QProscan(QFrame):

    def __init__(self):
        super(QProscan, self).__init__()
        self.instrument = pyproscan()
        self.initUI()
        atexit.register(self.shutdown)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.setInterval(200)

    def stop(self):
        self._timer.stop()

    def start(self):
        self._timer.start()
        return self

    def shutdown(self):
        self.stop()
        self.instrument.close()

    def initUI(self):
        self.setFrameShape(QFrame.Box)
        self.wx = self.counter_widget()
        self.wy = self.counter_widget()
        self.wz = self.counter_widget()
        self.bstop = QPushButton('Stop')
        self.bstop.clicked.connect(self.instrument.stop)
        layout = QGridLayout()
        layout.setMargin(1)
        layout.setHorizontalSpacing(1)
        layout.setVerticalSpacing(1)
        layout.addWidget(QLabel('Stage'), 1, 1, 1, 4)
        layout.addWidget(QLabel('x'), 2, 1)
        layout.addWidget(QLabel('y'), 2, 2)
        layout.addWidget(QLabel('z'), 2, 3)
        layout.addWidget(self.wx, 3, 1)
        layout.addWidget(self.wy, 3, 2)
        layout.addWidget(self.wz, 3, 3)
        layout.addWidget(self.bstop, 3, 4)
        self.setLayout(layout)

    def counter_widget(self):
        lcd = QLCDNumber(self)
        lcd.setNumDigits(9)
        lcd.setSegmentStyle(QLCDNumber.Flat)
        palette = lcd.palette()
        palette.setColor(palette.WindowText, QColor(0, 0, 0))
        palette.setColor(palette.Background, QColor(255, 255, 224))
        lcd.setPalette(palette)
        lcd.setAutoFillBackground(True)
        return lcd

    def update(self):
        position = self.instrument.position()
        if position is not None:
            self.wx.display(position[0])
            self.wy.display(position[1])
            self.wz.display(-position[2])  # NOTE: Special sign

    def setXOrigin(self):
        self.instrument.setPosition(x=0)

    def setYOrigin(self):
        self.instrument.setPosition(y=0)

    def setZOrigin(self):
        self.instrument.setPosition(z=0)


def main():
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    stage = QProscan()
    stage.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
