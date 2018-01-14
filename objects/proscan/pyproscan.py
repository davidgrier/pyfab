import serial
from serial.tools.list_ports import comports
import io


class pyproscan(object):

    def __init__(self):
        self.eol = '\r'
        self.timeout = 0.1
        self.findProscan()

    def close(self):
        self.ser.close()

    def findProscan(self):
        manufacturer = 'Prolific'
        ports = list(comports())
        if len(ports) <= 0:
            print('No serial ports found')
            return
        for port in ports:
            if port.manufacturer is None:
                continue
            if manufacturer not in port.manufacturer:
                continue
            self.ser = serial.Serial(port.device,
                                     baudrate=9600,
                                     bytesize=serial.EIGHTBITS,
                                     parity=serial.PARITY_NONE,
                                     stopbits=serial.STOPBITS_ONE,
                                     timeout=self.timeout)
            if not self.ser.isOpen():
                continue
            buffer = io.BufferedRWPair(self.ser, self.ser, 1)
            self.sio = io.TextIOWrapper(buffer, newline=self.eol,
                                        line_buffering=True)
            if self.identify():
                return
            self.ser.close()
        print('Proscan not found!')

    def identify(self):
        res = self.command('VERSION')
        return len(res) == 3

    def command(self, str):
        cmd = unicode(str + self.eol)
        self.sio.write(cmd)
        res = self.sio.readline().decode().strip()
        return res

    def stop(self):
        return self.command('I')

    def halt(self):
        return self.command('K')

    def position(self):
        return [int(x) for x in self.command('P').split(',')]

    def resolution(self):
        rx = self.command('RES,x')
        ry = self.command('RES,y')
        rz = self.command('RES,z')
        return(rx, ry, rz)


def main():
    a = pyproscan()
    print('position:', a.position())
    print('resolution:', a.resolution())


if __name__ == '__main__':
    main()
