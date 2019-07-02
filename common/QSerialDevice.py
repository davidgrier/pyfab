# -*- coding: utf-8 -*-

from PyQt5.QtSerialPort import (QSerialPort, QSerialPortInfo)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QSerialDevice(QSerialPort):

    def __init__(self, parent=None, info=None,
                 eol='\r',
                 manufacturer='Prolific',
                 baudrate=QSerialPort.Baud9600,
                 databits=QSerialPort.Data8,
                 parity=QSerialPort.NoParity,
                 stopbits=QSerialPort.OneStop,
                 timeout=1000):
        self.eol = eol
        self.manufacturer = manufacturer
        self.baudrate = baudrate
        self.databits = databits
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        if info is None:
            raise ValueError('Could not find serial device')
        super(QSerialDevice, self).__init__(info, parent)
        self.setBaudRate(self.baudrate)
        self.setDataBits(self.databits)
        self.setParity(self.parity)
        self.setStopBits(self.stopbits)
        self.open(QSerialPort.ReadWrite)

    def send(self, string):
        self.write(string.encode())

    def identify(self):
        return False


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    ports = QSerialPortInfo.availablePorts()
    if len(ports) < 1:
        logger.warning('No serial ports')
    for port in ports:
        info = QSerialPortInfo(port)
        print(info.systemLocation())
        a = QSerialDevice(info=info)
        if a.isOpen():
            print('open')
            a.write(b'VERSION')
            a.waitForReadyRead(a.timeout)
            print(a.readAll())
            a.close()
    sys.exit(app.exec_())
