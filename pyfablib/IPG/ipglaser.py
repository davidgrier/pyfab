# -*- coding: utf-8 -*-

"""Abstraction of an IPG fiber laser."""

from common.SerialDevice import SerialDevice
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


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

    def flags(self):
        return int(self.command('STA'))

    def flagSet(self, flagstr, flags=None):
        if not isinstance(flags, int):
            flags = self.flags()
        return bool(flags & self.flag[flagstr])

    def current(self):
        cur = float(self.command('RDC'))
        min = float(self.command('RNC'))
        set = float(self.command('RCS'))
        return cur, min, set

    def temperature(self):
        return float(self.command('RCT'))

    def keyswitch(self, flags=None):
        return not self.flagSet('KEY', flags)

    def startup(self, flags=None):
        return self.flagSet('EMS', flags)

    def emission(self, flags=flags, state=None):
        if state is True:
            res = self.command('EMON')
            return 'ERR' not in res
        if state is False:
            res = self.command('EMOFF')
            return 'ERR' not in res
        return self.flagSet('EMX', flags)

    def aimingbeam(self, state=None):
        if state is True:
            self.command('ABN')
        elif state is False:
            self.command('ABF')
        return self.flagSet('AIM')

    def error(self, flags=None):
        if not self.flagSet('ERR', flags):
            logger.info('No errors')
            return False
        if self.flagSet('TMP', flags):
            logger.warning('ERROR: Over-temperature condition')
        if self.flagSet('BKR', flags):
            logger.warning('ERROR: Excessive backreflection')
        if self.flagSet('PWR', flags):
            logger.warning('ERROR: Power supply off')
        if self.flagSet('UNX', flags):
            logger.warning('ERROR: Unexpected laser output')
        return True


def main():
    a = ipglaser()
    print(a.power())

    b = ipglaser()
    print(b.power())


if __name__ == '__main__':
    main()
