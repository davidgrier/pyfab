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

    # Overrides for QSerialDevice
    def identify(self):
        return len(self.version()) > 3

    def poll(self):
        '''Poll device for current status'''
        self.send('STA')  # status flags
        self.send('ROP')  # power

    @pyqtSlot(str)
    def process(self, msg):
        '''Process response from device poll'''
        part = msg.split()
        if len(part) < 2:
            logger.warning('Possible error: {}'.format(msg))
            return
        cmd = part[0]
        value = part[1]
        if 'STA' in cmd:
            state = int(value)
            status = (self.keyswitch(state),
                      self.aimingbeam(state),
                      self.startup(state) + self.emission(state),
                      self.error(state))
            self.sigStatus.emit(status)
        elif 'ROP' in cmd:
            power = self.power(value)
            self.sigPower.emit(power)

    # Instrument control
    def command(self, cmd):
        '''Handshake command synchronously and return response'''
        res = self.handshake(cmd)
        if cmd not in res:
            logger.warning('Possible error: {}'.format(res))
            return res
        parts = res.split()
        if len(parts) >= 2:
            return parts[1]
        return res

    @pyqtSlot(bool)
    def setAimingbeam(self, state=None):
        '''Control aiming laser'''
        if state is True:
            self.command('ABN')
        elif state is False:
            self.command('ABF')

    @pyqtSlot(bool)
    def setEmission(self, state=None):
        '''Control laser emission'''
        if state is True:
            res = self.command('EMON')
        elif state is False:
            res = self.command('EMOFF')

    # Properties
    def version(self):
        '''Instrument firmware version'''
        return self.command('RFV')

    def power(self, value=None):
        '''Laser emission power [W]'''
        if value is None:
            value = self.command('ROP')
        if 'Off' in value:
            power = 0.
        elif 'Low' in value:
            power = 0.1
        else:
            power = float(value)
        return power

    def current(self):
        '''Laser diode current [A]'''
        cur = float(self.command('RDC'))
        min = float(self.command('RNC'))
        set = float(self.command('RCS'))
        return cur, min, set

    def temperature(self):
        '''Laser temperature [C]'''
        return float(self.command('RCT'))

    # Status flags
    def flags(self):
        '''Instrument status flags'''
        return int(self.command('STA'))

    def flagSet(self, flagstr, flags=None):
        if not isinstance(flags, int):
            flags = self.flags()
        return bool(flags & self.flag[flagstr])

    def keyswitch(self, flags=None):
        return not self.flagSet('KEY', flags)

    def startup(self, flags=None):
        return self.flagSet('EMS', flags)

    def aimingbeam(self, flags=None):
        return self.flagSet('AIM')

    def emission(self, flags=None):
        return self.flagSet('EMX', flags)

    def error(self, flags=None):
        if flags is None:
            flags = self.flags()
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
    a = Ipglaser()
    print(a.power())

    b = Ipglaser()
    print(b.power())


if __name__ == '__main__':
    main()
