# -*- coding: utf-8 -*-

"""Control panel for an IPG fiber laser."""

from PyQt4 import QtCore, QtGui
import os
import numpy as np
from .ipglaser import ipglaser as ipg
import atexit


def led(name):
    led_size = 24
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'icons/' + name + '.png')
    return QtGui.QPixmap(filename).scaledToWidth(led_size)


class indicator(QtGui.QWidget):

    def __init__(self, title, states=None, button=False, **kwargs):
        super(indicator, self).__init__(**kwargs)

        self.title = title
        self.led_size = 24
        if states is None:
            states = [led('green-led-off'), led('green-led-on')]
        self.states = states
        self.init_ui(button)

    def init_ui(self, button):
        layout = QtGui.QVBoxLayout()
        layout.setMargin(0)
        layout.setSpacing(1)
        if button:
            self.button = QtGui.QPushButton(self.title, self)
            layout.addWidget(self.button)
        else:
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
        self.init_ui()

    def init_ui(self):
        self.setFrameStyle(QtGui.QFrame.Panel | QtGui.QFrame.Sunken)
        layout = QtGui.QHBoxLayout()
        self.led_key = indicator('keyswitch')
        self.led_aim = indicator('  aiming ',
                                 [led('amber-led-off'), led('amber-led-on')],
                                 button=True)
        self.led_aim.button.setToolTip('Toggle aiming laser on/off')
        self.led_emx = indicator(' emission',
                                 [led('red-led-off'),
                                  led('red-led-on'),
                                  led('amber-led-on')],
                                 button=True)
        self.led_emx.button.setToolTip('Toggle laser emission on/off')
        self.led_flt = indicator('  fault  ',
                                 [led('amber-led-off'), led('amber-led-on')])
        layout.setMargin(2)
        layout.setSpacing(1)
        layout.addWidget(self.led_key)
        layout.addWidget(self.led_aim)
        layout.addWidget(self.led_emx)
        layout.addWidget(self.led_flt)
        self.setLayout(layout)

    def update(self, key, aim, emx, flt):
        self.led_key.set(key)
        self.led_aim.set(aim)
        self.led_emx.set(emx)
        self.led_flt.set(flt)


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
        self._timer.setInterval(1000)

    def stop(self):
        self._timer.stop()

    def start(self):
        self._timer.start()
        return self

    def shutdown(self):
        self.stop()
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
        self.wstatus.led_aim.button.clicked.connect(self.toggleaim)
        self.wstatus.led_emx.button.clicked.connect(self.toggleemission)
        return w

    def toggleaim(self):
        state = self.instrument.aimingbeam()
        self.instrument.aimingbeam(state=not state)

    def toggleemission(self):
        state = self.instrument.emission()
        self.instrument.emission(state=not state)

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
