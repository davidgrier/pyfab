from ..SerialDevice import SerialDevice


class pyproscan(SerialDevice):

    def __init__(self):
        super(pyproscan, self).__init__()

    def identify(self):
        res = self.command('VERSION')
        return len(res) == 3

    def command(self, str):
        self.write(str)
        return self.readln()

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
