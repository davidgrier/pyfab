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
        if not self.find():
            raise ValueError('IPG Laser not found')

    def close(self):
        self.ser.close()

    def find(self):
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
                        self.ser.close()
                        continue
            except serial.SerialException as ex:
                print('Port {0} is unavailable: {1}'.format(port, ex))
                continue
            buffer = io.BufferedRWPair(self.ser, self.ser, 1)
            self.sio = io.TextIOWrapper(buffer, newline=self.eol,
                                        line_buffering=True)
            if self.identify():
                return True
            self.ser.close()
        print('IPG Laser not found!')
        return False

    def identify(self):
        res = self.command('RFV')
        return len(res) > 3

    def command(self, str):
        if not self.isready:
            return None
        cmd = unicode(str + self.eol)
        self.sio.write(cmd)
        res = self.sio.readline().decode().strip()
        print(res)
        return res

    def power(self):
        return self.command('ROP')


def main():
    a = ipglaser.find()
    print(a.power())

    b = ipglaser()
    print(b.power())


if __name__ == '__main__':
    main()
