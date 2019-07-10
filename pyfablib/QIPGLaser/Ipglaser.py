# -*- coding: utf-8 -*-

"""Abstraction of an IPG fiber laser."""

from PyQt5.QtCore import (pyqtSignal, pyqtSlot)
from common.QSerialDevice import QSerialDevice

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Ipglaser(QSerialDevice):

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

    sigStatus = pyqtSignal(object)
    sigPower = pyqtSignal(float)

    def __init__(self):
        super(Ipglaser, self).__init__(baudrate=57600)
        self.flag['ERR'] = (self.flag['TMP'] |
                            self.flag['BKR'] |
                            self.flag['PWR'] |
                            self.flag['UNX'])

    def identify(self):
        return len(self.version()) > 3

    def command(self, cmd):
        res = self.handshake(cmd)
        if cmd not in res:
            return cmd
        print(res)
        cmd, value = res.split()
        return value

    def version(self):
        return self.command('RFV')

    def power(self, value=None):
        if value is None:
            value = self.command('ROP')
        if 'Off' in value:
            power = 0.
        elif 'Low' in value:
            power = 0.1
        else:
            power = float(value)
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

    @pyqtSlot(bool)
    def emission(self, state=None):
        if state is True:
            res = self.command('EMON')
            return 'ERR' not in res
        if state is False:
            res = self.command('EMOFF')
            return 'ERR' not in res

    @pyqtSlot(bool)
    def aimingbeam(self, state=None):
        if state is True:
            self.command('ABN')
        elif state is False:
            self.command('ABF')

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

    def poll(self):
        self.send('STA')
        self.send('ROP')

    @pyqtSlot(str)
    def process(self, msg):
        part = msg.split()
        cmd = part[0]
        value = part[1]
        if 'STA' in cmd:
            status = int(value)
            state = (not self.flagSet('KEY', status),
                     self.flagSet('AIM', status),
                     (self.flagSet('EMS', status) +
                      self.flagSet('EMX', status)),
                     (self.flagSet('TMP', status) or
                      self.flagSet('BKR', status) or
                      self.flagSet('PWR', status) or
                      self.flagSet('UNX', status)))
            self.sigStatus.emit(state)
        elif 'ROP' in cmd:
            power = self.power(value)
            self.sigPower.emit(power)


def main():
    a = Ipglaser()
    print(a.power())

    b = Ipglaser()
    print(b.power())


if __name__ == '__main__':
    main()
