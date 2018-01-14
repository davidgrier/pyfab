from ..SerialDevice import SerialDevice
import numpy as np

class ipglaser(SerialDevice):

    def __init__(self):
	super(ipglaser, self).__init__(baudrate=57600)

    def identify(self):
        res = self.command('RFV')
        return len(res) > 3

    def command(self, str):
        self.write(str)
	res = self.readln()
	if str not in res:
	    return str
	res = res.replace(str, '').replace(': ', '')
        return res

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

    def status(self):
	res = self.command('STA')
	sta = np.uint32(res)
	status = dict()
	status['overtemperature'] = bool(sta & 2)
	status['emission'] = bool(sta & 4)
	status['highbackreflection'] = bool(sta & 8)
	status['analogcontrol'] = bool(sta & 16)
	status['modulesdisconnected'] = bool(sta & 64)
	status['modulesfailed'] = bool(sta & 128)
	status['aimingbeam'] = bool(sta & 256)
	status['powersupplyoff'] = bool(sta & 2048)
	status['modulationenabled'] = bool(sta & 4096)
	status['laserenable'] = bool(sta & 16384) 
	status['emissionstartup'] = bool(sta & 32768)
        status['keyswitchon'] = not bool(sta & 2097152)
	return status	

    def keyswitch(self):
	res = self.command('STA')
	sta = np.uint32(res)
	return not bool(sta & 2097152)

    def emission(self, state=None):
	if state is True:
	    res = self.command('EMON')
	    return 'ERR' not in res
	if state is False:
	    res = self.command('EMOFF')
	    return 'ERR' not in res
	res = self.command('STA')
	sta = np.uint32(res)
	return bool(sta & 4)



def main():
    a = ipglaser()
    print(a.power())

    b = ipglaser()
    print(b.power())


if __name__ == '__main__':
    main()
