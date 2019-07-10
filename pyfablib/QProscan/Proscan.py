# -*- coding: utf-8 -*-

'''Abstraction of a Prior Proscan stage controller'''

from PyQt5.QtCore import (pyqtSlot, pyqtSignal)
from common.QSerialDevice import QSerialDevice

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Proscan(QSerialDevice):

    updateX = pyqtSignal(int)
    updateY = pyqtSignal(int)
    updateZ = pyqtSignal(int)

    def __init__(self, parent, *args, **kwargs):
        super(Proscan, self).__init__(parent, *args, **kwargs)

    def identify(self):
        res = self.handshake('VERSION')
        print(len(res))
        return len(res) == 3

    @pyqtSlot()
    def poll(self):
        self.send('P')

    @pyqtSlot(str)
    def process(self, msg):
        if ',' not in msg:
            logger.warning('unexpected response: {}'.format(msg))
            return
        x, y, z = [int(val.strip()) for val in msg.split(',')]
        self.updateX.emit(x)
        self.updateY.emit(y)
        self.updateZ.emit(z)


def main():
    a = Proscan()
