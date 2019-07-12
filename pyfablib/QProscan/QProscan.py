# -*- coding: utf-8 -*-

'''Abstraction of a Prior Proscan stage controller'''

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import (pyqtSlot, QTimer)
from .QProscan_UI import Ui_QProscan
from .Proscan import Proscan

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class QProscan(QWidget, Ui_QProscan):

    def __init__(self, parent=None, interval=200, **kwargs):
        super(QProscan, self).__init__(parent)
        self.setupUi(self)
        self.device = Proscan(self, **kwargs)
        self.device.updateX.connect(self.lcdX.display)
        self.device.updateY.connect(self.lcdY.display)
        self.device.updateZ.connect(self.lcdZ.display)
        self.timer = QTimer()
        self.timer.timeout.connect(self.device.poll)
        self.timer.setInterval(interval)

    def start(self):
        self.timer.start()

    def stop(self):
        self.timer.stop()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    wid = QProscan()
    wid.start()
    wid.show()
    sys.exit(app.exec_())
