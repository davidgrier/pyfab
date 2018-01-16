from PyQt4 import QtCore, QtGui
import os
import numpy as np
from .ipglaser import ipglaser as ipg
import atexit


class indicator(QtGui.QWidget):

    def __init__(self, title, states, **kwargs):
        super(indicator, self).__init__(**kwargs)
        self.title = title
        self.states = states
        self.init_ui()

    def init_ui(self):
        layout = QtGui.QVBoxLayout()
        layout.setMargin(0)
        layout.setSpacing(1)
        w = QtGui.QLabel(self.title)
        w.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(w)
        self.led = QtGui.QLabel()
        self.led.setAlignment(QtCore.Qt.AlignCenter)
        self.led.setPixmap(self.states[0])
        layout.addWidget(self.led)
        self.setLayout(layout)

    def set(self, state):
        self.led.setPixmap(self.states[state])


class status_widget(QtGui.QFrame):

    def __init__(self):
        super(status_widget, self).__init__()
        self.dir = os.path.dirname(__file__)
        self.led_size = 16
        self.init_ui()
        self.status(ipg.flag['AIM'])  # FIXME -- display purposes

    def init_ui(self):
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        green_on = self.led_pixmap('green-led-on').scaledToWidth(self.led_size)
        green_off = self.led_pixmap(
            'green-led-off').scaledToWidth(self.led_size)
        amber_on = self.led_pixmap('amber-led-on').scaledToWidth(self.led_size)
        amber_off = self.led_pixmap(
            'amber-led-off').scaledToWidth(self.led_size)
        red_on = self.led_pixmap('red-led-on').scaledToWidth(self.led_size)
        red_off = self.led_pixmap('red-led-off').scaledToWidth(self.led_size)
        layout = QtGui.QHBoxLayout()
        self.led_key = indicator('keyswitch', [green_off, green_on])
        self.led_aim = indicator('  aiming ', [amber_off, amber_on])
        self.led_emx = indicator(' emission', [red_off, red_on, amber_on])
        self.led_flt = indicator('  fault  ', [amber_off, amber_on])
        layout.setMargin(2)
        layout.setSpacing(1)
        layout.addWidget(self.led_key)
        layout.addWidget(self.led_aim)
        layout.addWidget(self.led_emx)
        layout.addWidget(self.led_flt)
        self.setLayout(layout)

    def led_pixmap(self, name):
        filename = os.path.join(self.dir, 'icons/' + name + '.png')
        w = QtGui.QPixmap(filename)
        return w

    def update(self, key, aim, emx, flt):
        self.led.key.set(key)
        self.led.aim.set(aim)
        self.led.emx.set(emx)
        self.led.flt.set(flt)

        
class power_widget(QtGui.QWidget):

    def __init__(self, **kwargs):
        super(power_widget, self).__init__(**kwargs)
        self.min = 0.
        self.max = 10.
        self.init_ui()
        self.value = 0

    def init_ui(self):
        layout = QtGui.QVBoxLayout()
        layout.setMargin(2)
        layout.setSpacing(1)
        title = QtGui.QLabel('power [W]')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        v = QtGui.QDoubleValidator(self.min, self.max, 4.)
        v.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.wvalue = QtGui.QLineEdit()
        self.wvalue.setValidator(v)
        self.wvalue.setAlignment(QtCore.Qt.AlignRight)
        self.wvalue.setMaxLength(6)
        self.wvalue.setReadOnly(True)
        layout.addWidget(self.wvalue)
        self.setLayout(layout)

    @property
    def value(self):
        self._value

    @value.setter
    def value(self, _value):
        value = np.clip(float(_value), self.min, self.max)
        self._value = value
        self.wvalue.setText(QtCore.QString('%.4f' % value))


class QIPGLaser(QtGui.QFrame):

    def __init__(self):
        super(QIPGLaser, self).__init__()
        self.instrument = ipg()
        self.init_ui()
        atexit.register(self.shutdown)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.setInverval(1000)
        self._timer.start()

    def shutdown(self):
        self._timer.stop()
        self.instrument.close()
        
    def init_ui(self):
        self.setFrameShape(QtGui.QFrame.Box)
        layout = QtGui.QVBoxLayout()
        layout.setMargin(0)
        layout.setSpacing(0)
        layout.addWidget(QtGui.QLabel(' Trapping Laser'))
        layout.addWidget(self.display_widget())
        self.setLayout(layout)

    def display_widget(self):
        self.wstatus = status_widget()
        self.wpower = power_widget()
        w = QtGui.QWidget()
        layout = QtGui.QHBoxLayout()
        layout.setSpacing(1)
        layout.addWidget(self.wstatus)
        layout.addWidget(self.wpower)
        w.setLayout(layout)
        return w

    def update(self):
        flags = self.instrument.flags()
        self.wstatus.update(self.instrument.keyswitch(flags),
                            self.instrument.aimingbeam(flags),
                            (self.instrument.startup(flags) +
                             self.instrument.emission(flags)),
                            self.instrument.error(flags))
        self.wpower.value = self.instrument.power()


def main():
    import sys

    app = QtGui.QApplication(sys.argv)
    w = status_widget()
    w.show()
    sys.exit(app.exec_())
