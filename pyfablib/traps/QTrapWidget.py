# -*- coding: utf-8 -*-

"""Control panel for trap properties."""

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, Qt, QRegExp)
from PyQt5.QtWidgets import (QWidget, QFrame, QLineEdit, QLabel,
                             QScrollArea,
                             QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import (QDoubleValidator, QRegExpValidator)
from .QTrap import QTrap
import numpy as np


class QTrapPropertyEdit(QLineEdit):

    """Control for one property of one trap"""

    valueChanged = pyqtSignal(object, float)

    def __init__(self, name, value, decimals=1):
        super(QTrapPropertyEdit, self).__init__()
        self.setAlignment(Qt.AlignRight)
        self.setFixedWidth(50)
        self.setMaxLength(8)
        self.fmt = '%.{}f'.format(decimals)
        v = QDoubleValidator(decimals=decimals)
        v.setNotation(QDoubleValidator.StandardNotation)
        self.setValidator(v)
        self.name = name
        self.value = value
        self.returnPressed.connect(self.updateValue)

    @pyqtSlot()
    def updateValue(self):
        self.value = float(str(self.text()))
        self.valueChanged.emit(self.name, self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.setText(self.fmt.format(value))
        self._value = value


class QTrapListPropertyEdit(QLineEdit):

    """Control for one list-like property of one trap"""

    valueChanged = pyqtSignal(object, object)

    def __init__(self, name, value):
        super(QTrapListPropertyEdit, self).__init__()
        self.setAlignment(Qt.AlignRight)
        self.setFixedWidth(50)
        numberrx = '([+-]?\d+\.?\d*)'
        listrx = '\[' + '(?:\s*' + numberrx + '\s*,)*\s*' + numberrx + '\s*\]'
        print(listrx)
        self.rx = QRegExp(listrx)
        val = QRegExpValidator(self.rx)
        self.setValidator(val)
        self.name = name
        self.value = value
        self.returnPressed.connect(self.updateValue)

    @pyqtSlot()
    def updateValue(self):
        txt = str(self.text())
        self.value = np.fromstring(txt[1:-1], sep=',', dtype=np.float)
        self.valueChanged.emit(self.name, self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.setText(str(value))
        self._value = value


class QTrapPropertyWidget(QWidget):

    """Control for properties of one trap."""

    def __init__(self, trap):
        super(QTrapPropertyWidget, self).__init__()
        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignLeft)
        self.wid = dict()
        for name in trap.properties.keys():
            self.wid[name] = self.propertyWidget(trap, name)
            tip = trap.__class__.__name__ + ': ' + name
            self.wid[name].setStatusTip(tip)
            if trap.properties[name]['tooltip']:
                self.wid[name].setToolTip(name)
            layout.addWidget(self.wid[name])
        trap.valueChanged.connect(self.updateValues)
        self.setLayout(layout)

    def propertyWidget(self, trap, name):
        value = getattr(trap, name)
        decimals = trap.properties[name]['decimals']
        if isinstance(value, list):
            wid = QTrapListPropertyEdit(name, value)
        else:
            wid = QTrapPropertyEdit(name, value, decimals=decimals)
        wid.valueChanged.connect(trap.setProperty)
        return wid

    @pyqtSlot(QTrap)
    def updateValues(self, trap):
        for name in trap.properties.keys():
            value = getattr(trap, name)
            self.wid[name].value = value


class QTrapWidget(QFrame):

    """Controls for all traps."""

    def __init__(self, pattern=None):
        super(QTrapWidget, self).__init__()
        self.properties = dict()
        self.init_ui()
        if pattern is not None:
            pattern.trapAdded.connect(self.registerTrap)

    def init_ui(self):
        self.setFrameShape(QFrame.Box)
        inner = QWidget()
        self.layout = QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setAlignment(Qt.AlignTop)
        inner.setLayout(self.layout)
        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.layout.addWidget(self.labelLine())

    def labelLine(self):
        widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignLeft)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        for name in ['x', 'y', 'z', 'alpha', 'phi']:
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            label.setFixedWidth(50)
            layout.addWidget(label)
        widget.setLayout(layout)
        return widget

    def registerTrap(self, trap):
        trapWidget = QTrapPropertyWidget(trap)
        self.properties[trap] = trapWidget
        self.layout.addWidget(trapWidget)
        trap.destroyed.connect(lambda: self.unregisterTrap(trap))

    def unregisterTrap(self, trap):
        self.properties[trap].deleteLater()
        del self.properties[trap]

    def count(self):
        return self.layout.count()


if __name__ == '__main__':
    import sys
    from QTrap import QTrap
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    wtrap = QTrapWidget()
    trapa = QTrap()
    trapb = QTrap()
    trapc = QTrap()
    wtrap.registerTrap(trapa)
    wtrap.registerTrap(trapb)
    wtrap.show()
    # change trap properties
    trapa.r = (100, 100, 10)
    trapc.r = (50, 50, 5)
    # remove trap after display
    trapb.deleteLater()
    wtrap.registerTrap(trapc)

    sys.exit(app.exec_())
