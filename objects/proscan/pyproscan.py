from ..SerialDevice import SerialDevice


class pyproscan(SerialDevice):

    def __init__(self):
        super(pyproscan, self).__init__()
        self.vmax = self.maxSpeed()
        self.vzmax = self.maxZSpeed()
        self.a = self.acceleration()
        self.az = self.zAcceleration()
        self.s = self.scurve()
        self.sz = self.zScurve()

    def identify(self):
        res = self.command('VERSION')
        return len(res) == 3

    def command(self, str):
        self.write(str)
        return self.readln()

    # Stage motion controls
    def stop(self):
        '''Stop all motion
        '''
        return self.command('I')

    def halt(self):
        '''Emergency stop
        '''
        return self.command('K')

    def moveTo(self, x, y, z=None):
        '''Move stage to absolute position (x, y, z) [um]
        '''
        if z is None:
            self.command('G,%d,%d' % (x, y))
        else:
            self.command('G,%d,%d,%d' % (x, y, z))

    def moveBy(self, dx, dy, dz=None):
        '''Displace stage by specified amount (dx, dy, dz) [um]
        '''
        if dz is None:
            self.command('GR,%d,%d' % (dx, dy))
        else:
            self.command('GR,%d,%d,%d' & (dx, dy, dz))

    def moveX(self, x):
        '''Move stage along x axis to specified position [um]
        '''
        self.command('GX,%d', x)

    def moveY(self, y):
        '''Move stage along y axis to specified position [um]
        '''
        self.command('GY,%d', y)

    def moveZ(self, z):
        '''Move stage along z axis to specified position [um]
        '''
        self.command('GZ,%d', z)

    # Properties of motion controller
    def position(self):
        '''Return current position of stage [um]
        '''
        return [int(x) for x in self.command('P').split(',')]

    def setPosition(self, x=None, y=None, z=None):
        '''Define coordinates for current stage position
        '''
        if x is not None:
            if len(x) == 3:
                y = x[1]
                z = x[2]
                x = x[0]
            self.command('PX,%d' % x)
        if y is not None:
            self.command('PY,%d' % y)
        if z is not None:
            self.command('PZ,%d' % z)

    def resolution(self):
        '''Return resolution of stage motion [um]
        '''
        rx = self.command('RES,x')
        ry = self.command('RES,y')
        rz = self.command('RES,z')
        return(rx, ry, rz)

    def maxSpeed(self):
        '''Return maximum speed of in-plane motions [um/s]
        '''
        return self.command('SMS')

    def setMaxSpeed(self, speed):
        '''Set maximum in-plane speed [um/s]
        '''
        self.command('SMS,%d' % speed)

    def maxZSpeed(self):
        '''Return maximum axial speed [um/s]
        '''
        return self.command('SMZ')

    def setMaxZSpeed(self, speed):
        '''Set maximum axial speed [um/s]
        '''
        self.command('SMZ,%d' % speed)

    def acceleration(self):
        '''Return in-plane acceleration [um/s^2]
        '''
        return self.command('SAS')

    def setAcceleration(self, acceleration):
        '''Set in-plane acceleration [um/s^2]
        '''
        self.command('SAS,%d' % acceleration)

    def zAcceleration(self):
        '''Return axial acceleration [um/s^2]
        '''
        return self.command('SAZ')

    def setZAcceleration(self, acceleration):
        '''Set axial acceleration [um/s^2]
        '''
        self.command('SAZ,%d' % acceleration)

    def scurve(self):
        '''Return in-plane s-curve value [um/s^3]
        '''
        return self.command('SCS')

    def setSCurve(self, scurve):
        '''Set in-plane s-curve value [um/s^3]
        '''
        self.command('SCS,%d' % scurve)

    def zScurve(self):
        '''Return axial s-curve value [um/s^3]
        '''
        return self.command('SCZ')

    def setZSCurve(self, scurve):
        '''Set axial s-curve value [um/s^3]
        '''
        self.command('SCZ,%d' % scurve)

    def reset(self):
        '''Reset motion controls to starting values
        '''
        self.setMaxSpeed(self.vmax)
        self.setMaxZSpeed(self.vzmax)
        self.setAcceleration(self.a)
        self.setZAcceleration(self.az)
        self.setSCurve(self.s)
        self.setZSCurve(self.sz)


def main():
    a = pyproscan()
    print('position:', a.position())
    print('resolution:', a.resolution())


if __name__ == '__main__':
    main()
