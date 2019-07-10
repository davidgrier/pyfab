# -*- coding: utf-8 -*-

'''Abstraction of a Prior Proscan stage controller'''

from common.QSerialDevice import QSerialDevice
from PyQt5.QtCore import pyqtSlot

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Proscan(QSerialDevice):

    def __init__(self):
        super(Proscan, self).__init__()

    def identify(self):
        res = self.handshake('VERSION')
        return len(res) == 3

    @pyqtSlot(str)
    def process(self, msg):
        print(msg)
