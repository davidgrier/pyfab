# -*- coding: utf-8 -*-

from PyQt5.QtCore import (pyqtSlot, QByteArray)
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
        self.readyRead.connect(self.receive)
        self.buffer = QByteArray()
        if port is None:
            self.find()
        else:
            self.setup(port)
        if not self.isOpen():
            raise ValueError('Could not find serial device')

    def setup(self, portinfo):
        logger.debug('Setting up')
        if portinfo is None:
            logger.info('No serial port specified')
            return False
        name = portinfo.systemLocation()
        if portinfo.isBusy():
            logger.info('Port is busy: {}'.format(name))
            return False
        self.setPort(portinfo)
        self.setBaudRate(self.baudrate)
        self.setDataBits(self.databits)
        self.setParity(self.parity)
        self.setStopBits(self.stopbits)
        if not self.open(QSerialPort.ReadWrite):
            logger.info('Could not open port: {}'.format(name))
            return False
        if self.identify():
            logger.info('Device found at {}'.format(name))
            return True
        self.close()
        logger.info('Device not connected to {}'.format(name))
        return False

    def find(self):
        ports = QSerialPortInfo.availablePorts()
        if len(ports) < 1:
            logger.warning('No serial ports detected')
            return
        for port in ports:
            portinfo = QSerialPortInfo(port)
            if self.setup(portinfo):
                break

    def identify(self):
        return True

    def process(self, data):
        logger.debug('received: {}'.format(data))

    def send(self, data):
        cmd = data + self.eol
        self.write(cmd.encode())

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
        self.buffer.append(self.readAll())
        if self.buffer.contains(self.eol.encode()):
            self.process(bytes(self.buffer).decode())
            self.buffer.clear()

    def handshake(self, cmd):
        self.blockSignals(True)
        self.send(cmd)
        if self.waitForReadyRead(self.timeout):
            res = self.gets()
        else:
            res = ''
        self.blockSignals(False)
        return res


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
