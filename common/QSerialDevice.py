# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtSerialPort import (QSerialPort, QSerialPortInfo)
from PyQt5.QtWidgets import QMainWindow

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QSerialDevice(QSerialPort):

    def __init__(self, parent=None, port=None,
                 eol='\r',
                 manufacturer='Prolific',
                 baudrate=QSerialPort.Baud9600,
                 databits=QSerialPort.Data8,
                 parity=QSerialPort.NoParity,
                 stopbits=QSerialPort.OneStop,
                 timeout=1000):
        super(QSerialDevice, self).__init__(parent=parent)
        self.eol = eol
        self.manufacturer = manufacturer
        self.baudrate = baudrate
        self.databits = databits
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        # self.readyRead.connect(self.receive)
        if port is not None:
            self.setup(port)

    def setup(self, port):
        print('setting up')
        self.setPort(port)
        self.setBaudRate(self.baudrate)
        self.setDataBits(self.databits)
        self.setParity(self.parity)
        self.setStopBits(self.stopbits)
        if not self.open(QSerialPort.ReadWrite):
            raise ValueError('Could not open serial device')

    def getc(self):
        return bytes(self.read(1)).decode('utf8')
    
    def gets(self):
        str = ''
        char = self.getc()
        while char != self.eol:
            str += char
            if self.waitForReadyRead(self.timeout):
                char = self.getc()
            else:
                break
        return str

    @pyqtSlot()
    def receive(self):
        print('reading')
        print(self.gets())

    def send(self, data):
        print('sending')
        cmd = data + self.eol
        nsent = self.write(cmd.encode())
        print(nsent)

    def handshake(self, cmd):
        self.send(cmd)
        if self.waitForReadyRead(self.timeout):
            return self.gets()
        else:
            return None

    def identify(self):
        return False


class Main(QMainWindow):
    def __init__(self, number=0):
        super().__init__()
        ports = QSerialPortInfo.availablePorts()
        if len(ports) < 1:
            logger.warning('No serial ports')
        port = QSerialPortInfo(ports[number])
        print(port.systemLocation())
        self.serial = QSerialDevice(port=port)
        if self.serial.isOpen():
            print('open')
            print(self.serial.handshake('VERSION'))

    def closeEvent(self, event):
        self.serial.close()


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    gui = Main(0)
    gui.show()
    sys.exit(app.exec_())
