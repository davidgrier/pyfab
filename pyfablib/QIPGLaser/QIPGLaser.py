# -*- coding: utf-8 -*-

"""Control panel for an IPG fiber laser."""

from PyQt5.QtCore import (Qt, QTimer, pyqtSlot, pyqtProperty)
from PyQt5.QtGui import (QPixmap, QDoubleValidator)
from PyQt5.QtWidgets import (QWidget, QFrame, QPushButton, QLabel,
                             QLineEdit, QHBoxLayout, QVBoxLayout)
import os
import numpy as np
from .Ipglaser import Ipglaser


def led(name):
    led_size = 24
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, 'icons/' + name + '.png')
    return QPixmap(filename).scaledToWidth(led_size)


class Indicator(QWidget):

    def __init__(self, title,
                 states=None,
                 button=False, **kwargs):
        super(Indicator, self).__init__(**kwargs)

        self.title = title
        self.led_size = 24
        if states is None:
            states = [led('green-led-off'), led('green-led-on')]
        self.states = states
        self.initUi(button)

    def initUi(self, button):
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

    @pyqtSlot(int)
    def setState(self, state):
        self._state = state
        self.led.setPixmap(self.states[state])

    @pyqtProperty(int)
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        self.setState(state)


class StatusWidget(QFrame):

    def __init__(self):
        super(StatusWidget, self).__init__()
        self.initUi()

    def initUi(self):
        self.setFrameStyle(QFrame.Panel | QFrame.Sunken)
        layout = QHBoxLayout()
        amber = [led('amber-led-off'), led('amber-led-on')]
        tricolor = [led('red-led-off'),
                    led('red-led-on'),
                    led('amber-led-on')]
        self.keyswitch = Indicator('keyswitch')
        self.aiming = Indicator('  aiming ', amber, button=True)
        self.aiming.button.setToolTip('Toggle aiming laser on/off')
        self.emission = Indicator(' emission', tricolor, button=True)
        self.emission.button.setToolTip('Toggle laser emission on/off')
        self.fault = Indicator('  fault  ', amber)

        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        layout.addWidget(self.keyswitch)
        layout.addWidget(self.aiming)
        layout.addWidget(self.emission)
        layout.addWidget(self.fault)
        self.setLayout(layout)

    @pyqtSlot(object)
    def update(self, status):
        key, aim, emx, flt = status
        self.keyswitch.setState(key)
        self.aiming.setState(aim)
        self.emission.setState(emx)
        self.fault.setState(flt)


class PowerWidget(QWidget):

    def __init__(self, **kwargs):
        super(PowerWidget, self).__init__(**kwargs)
        self.min = 0.
        self.max = 10.
        self.initUi()
        self.value = 0

    def initUi(self):
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

    @pyqtSlot(float)
    def setValue(self, value):
        value = np.clip(float(value), self.min, self.max)
        self._value = value
        self.wvalue.setText('{0:.4f}'.format(value))

    @pyqtProperty(float)
    def value(self):
        self._value

    @value.setter
    def value(self, value):
        self.setValue(value)


class QIPGLaser(QFrame):

    def __init__(self):
        super(QIPGLaser, self).__init__()
        self.instrument = Ipglaser()

        self.status = StatusWidget()
        self.power = PowerWidget()
        self.timer = QTimer(self)
        self.timer.setInterval(500)

        self.initUi()
        self.connectSignals()

    def initUi(self):
        self.setFrameShape(QFrame.Box)
        hlayout = QHBoxLayout()
        self.setLayout(hlayout)
        hlayout.setSpacing(1)
        hlayout.addWidget(self.status)
        hlayout.addWidget(self.power)

    def connectSignals(self):
        self.instrument.sigStatus.connect(self.status.update)
        self.instrument.sigPower.connect(self.power.setValue)
        self.status.aiming.button.clicked.connect(self.toggleaim)
        self.status.emission.button.clicked.connect(self.toggleemission)
        self.timer.timeout.connect(self.instrument.poll)

    def stop(self):
        self.timer.stop()

    def start(self):
        self.timer.start()
        return self

    def shutdown(self):
        self.stop()
        self.instrument.close()

    @pyqtSlot()
    def toggleaim(self):
        newstate = not self.status.aiming.state
        self.instrument.setAimingbeam(newstate)

    @pyqtSlot()
    def toggleemission(self):
        newstate = not self.status.emission.state
        self.instrument.setEmission(newstate)


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
