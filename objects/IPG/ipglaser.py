import fcntl
import serial
import io
from serial.tools.list_ports import comports


class ipglaser(object):

    def __init__(self):
        self.eol = '\r'
        self.manufacturer = 'Prolific'
        self.timeout = 0.1
        self.baudrate = 57600
        self.bytesize = serial.EIGHTBITS
        self.parity = serial.PARITY_NONE
        self.stopbits = serial.STOPBITS_ONE
        self.findIPGLaser()

    def close(self):
        self.ser.close()

    def findIPGLaser(self):
        ports = list(comports())
        if len(ports) <= 0:
            print('No serial ports found')
            return
        for port in ports:
            if port.manufacturer is None:
                continue
            if self.manufacturer not in port.manufacturer:
                continue
            try:
                self.ser = serial.Serial(port.device,
                                         baudrate=self.baudrate,
                                         bytesize=self.bytesize,
                                         parity=self.parity,
                                         stopbits=self.stopbits,
                                         timeout=self.timeout)
                if self.ser.isOpen():
                    try:
                        fcntl.flock(self.ser.fileno(),
                                    fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except IOError:
                        print 'Port {0} is busy'.format(self.ser.port)
                        continue
            except serial.SerialException as ex:
                print('Port {0} is unavailable: {1}'.format(port, ex))
                continue
            buffer = io.BufferedRWPair(self.ser, self.ser, 1)
            self.sio = io.TextIOWrapper(buffer, newline=self.eol,
                                        line_buffering=True)
            if self.identify():
                return
            self.ser.close()
        print('IPG Laser not found!')

    def identify(self):
        res = self.command('RFV')
        return len(res) > 3

    def command(self, str):
        cmd = unicode(str + self.eol)
        self.sio.write(cmd)
        res = self.sio.readline().decode().strip()
        print(res)
        return res

    def power(self):
        return self.command('ROP')


def main():
    a = ipglaser()
    b = ipglaser()
    print(a.power())
    print(b.power())


if __name__ == '__main__':
    main()
