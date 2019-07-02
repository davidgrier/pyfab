# -*- coding: utf-8 -*-

from PyQt5.QtSerialPort import (QSerialPort, QSerialPortInfo)

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QSerialDevice(QSerialPort):

    def __init__(self, parent=None,
                 eol='\r',
                 manufacturer='Prolific',
                 baudrate=QSerialPort.Baud9600,
                 databits=QSerialPort.Data8,
                 parity=QSerialPort.NoParity,
                 stopbits=QSerialPort.OneStop,
                 timeout=0.1):
        self.eol = eol
        self.manufacturer = manufacturer
        self.baudrate = baudrate
        self.databits = databits
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        info = self.find()
        if info is None:
            raise ValueError('Could not find serial device')
        super(QSerialDevice, self).__init__(info, parent)
        self.configurePort(self)

    def configurePort(self, port):
        port.setBaudRate(self.baudrate)
        port.setDataBits(self.databits)
        port.setParity(self.parity)
        port.setStopBits(self.stopbits)

    def find(self):
        ports = QSerialPortInfo.availablePorts()
        if len(ports) <= 0:
            logger.warning('No serial ports identified')
            return None
        for port in ports:
            info = QSerialPortInfo(port)
            if info.isBusy():
                continue
            logger.debug(info.systemLocation())
            serial = QSerialPort(info)
            self.configurePort(serial)
            found = self.identify(serial)
            serial.close()
            if found:
                return info
        return None

    def identify(self, port):
        return False

    def write(self, str):
        self.sio.write(str + self.eol)

    def readln(self):
        return self.sio.readline().strip()

    def available(self):
        return self.ser.in_waiting


if __name__ == '__main__':
    a = QSerialDevice()
