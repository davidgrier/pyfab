from ..SerialDevice import SerialDevice


class ipglaser(SerialDevice):

    flag = {'ERR': 0x1,
            'TMP': 0x2,       # TMP: ERROR: over-temperature condition
            'EMX': 0x4,       # EMX: laser emission
            'BKR': 0x8,       # BKR: ERROR: excessive backreflection
            'ACL': 0x10,      # ACL: analog control mode enabled
            'MDC': 0x40,      # MDC: module communication disconnected
            'MFL': 0x80,      # MFL: module(s) have failed
            'AIM': 0x100,     # AIM: aiming beam on
            'PWR': 0x800,     # PWR: ERROR: power supply off
            'MOD': 0x1000,    # MOD: modulation enabled
            'ENA': 0x4000,    # ENA: laser enable is asserted
            'EMS': 0x8000,    # EMS: emission startup
            'UNX': 0x20000,   # UNX: ERROR: unexpected emission detected
            'KEY': 0x200000}  # KEY: keyswitch in REM position

    def __init__(self):
        super(ipglaser, self).__init__(baudrate=57600)
        self.flag['ERR'] = (self.flag['TMP'] |
                            self.flag['BKR'] |
                            self.flag['PWR'] |
                            self.flag['UNX'])

    def identify(self):
        return len(self.version()) > 3

    def command(self, str):
        self.write(str)
        res = self.readln()
        if str not in res:
            return str
        res = res.replace(str, '').replace(': ', '')
        return res

    def version(self):
        return self.command('RFV')

    def power(self):
        res = self.command('ROP')
        if 'Off' in res:
            power = 0.
        elif 'Low' in res:
            power = 0.1
        else:
            power = float(res)
        return power

    def current(self):
        cur = float(self.command('RDC'))
        min = float(self.command('RNC'))
        set = float(self.command('RCS'))
        return cur, min, set

    def temperature(self):
        return float(self.command('RCT'))

    def keyswitch(self):
        flags = int(self.command('STA'))
        return not bool(flags & self.flag['KEY'])

    def startup(self):
        flags = int(self.command('STA'))
        return bool(flags & self.flag['EMS'])

    def emission(self, state=None):
        if state is True:
            res = self.command('EMON')
            return 'ERR' not in res
        if state is False:
            res = self.command('EMOFF')
            return 'ERR' not in res
        flags = int(self.command('STA'))
        return bool(flags & self.flag['EMX'])

    def aimingbeam(self, state=None):
        if state is True:
            self.command('ABN')
        elif state is False:
            self.command('ABF')
        flags = int(self.command('STA'))
        return bool(flags & self.flag['AIM'])

    def error(self, flags):
        if not bool(flags & self.flags['ERR']):
            print 'No errors'
            return
        if bool(flags & self.flags['TMP']):
            print 'ERROR: Over-temperature condition'
        if bool(flags & self.flags['BKR']):
            print 'ERROR: Excessive backreflection'
        if bool(flags & self.flags['PWR']):
            print 'ERROR: Power supply off'
        if bool(flags & self.flags['UNX']):
            print 'ERROR: Unexpected laser output'


def main():
    a = ipglaser()
    print(a.power())

    b = ipglaser()
    print(b.power())


if __name__ == '__main__':
    main()
