# -*- coding: utf-8 -*-

"""Control panel for an IPG fiber laser."""

from PyQt5.QtCore import (Qt, QTimer, pyqtSlot)
from PyQt5.QtGui import (QPixmap, QDoubleValidator)
from PyQt5.QtWidgets import (QWidget, QFrame, QPushButton, QLabel,
                             QLineEdit, QHBoxLayout, QVBoxLayout)
import os
import numpy as np
from Ipglaser import Ipglaser


def led(name):
    led_size = 24
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'icons/' + name + '.png')
    return QPixmap(filename).scaledToWidth(led_size)


class Indicator(QWidget):

    def __init__(self, title, states=None, button=False, **kwargs):
        super(Indicator, self).__init__(**kwargs)

        self.title = title
        self.led_size = 24
        if states is None:
            states = [led('green-led-off'), led('green-led-on')]
        self.states = states
        self.init_ui(button)

    def init_ui(self, button):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        if button:
            self.button = QPushButton(self.title, self)
            layout.addWidget(self.button)
        else:
            w = QLabel(self.title)
            w.setAlignment(Qt.AlignCenter)
            layout.addWidget(w)
        self.led = QLabel()
        self.led.setAlignment(Qt.AlignCenter)
        self.led.setPixmap(self.states[0])
        layout.addWidget(self.led)
        self.setLayout(layout)

    def set(self, state):
        self.led.setPixmap(self.states[state])


class StatusWidget(QFrame):

    def __init__(self):
        super(StatusWidget, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        layout = QHBoxLayout()
        self.led_key = Indicator('keyswitch')
        self.led_aim = Indicator('  aiming ',
                                 [led('amber-led-off'), led('amber-led-on')],
                                 button=True)
        self.led_aim.button.setToolTip('Toggle aiming laser on/off')
        self.led_emx = Indicator(' emission',
                                 [led('red-led-off'),
                                  led('red-led-on'),
                                  led('amber-led-on')],
                                 button=True)
        self.led_emx.button.setToolTip('Toggle laser emission on/off')
        self.led_flt = Indicator('  fault  ',
                                 [led('amber-led-off'), led('amber-led-on')])
        layout.setContentsMargins(2, 2, 2, 2)
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


class PowerWidget(QWidget):

    def __init__(self, **kwargs):
        super(PowerWidget, self).__init__(**kwargs)
        self.min = 0.
        self.max = 10.
        self.init_ui()
        self.value = 0

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        title = QLabel('power [W]')
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        v = QDoubleValidator(self.min, self.max, 4.)
        v.setNotation(QDoubleValidator.StandardNotation)
        self.wvalue = QLineEdit()
        self.wvalue.setValidator(v)
        self.wvalue.setAlignment(Qt.AlignRight)
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
        self.wvalue.setText('{0:.4f}'.format(value))


class QIPGLaser(QFrame):

    def __init__(self):
        super(QIPGLaser, self).__init__()
        self.instrument = Ipglaser()
        self.init_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self.instrument.poll)
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
        self.setFrameShape(QFrame.Box)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(QLabel(' Trapping Laser'))
        layout.addWidget(self.display_widget())
        self.setLayout(layout)

    def display_widget(self):
        self.wstatus = StatusWidget()
        self.wpower = PowerWidget()
        w = QWidget()
        layout = QHBoxLayout()
        layout.setSpacing(1)
        layout.addWidget(self.wstatus)
        layout.addWidget(self.wpower)
        w.setLayout(layout)
        self.wstatus.led_aim.button.clicked.connect(self.toggleaim)
        self.wstatus.led_emx.button.clicked.connect(self.toggleemission)
        return w

    @pyqtSlot()
    def toggleaim(self):
        state = self.instrument.aimingbeam()
        self.instrument.aimingbeam(state=not state)

    @pyqtSlot()
    def toggleemission(self):
        state = self.instrument.emission()
        self.instrument.emission(state=not state)

    @pyqtSlot()
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
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    wid = QIPGLaser()
    wid.start()
    wid.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
