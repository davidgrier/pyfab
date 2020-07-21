# -*- coding: utf-8 -*-

"""Control panel for trap properties."""

from PyQt5.QtCore import (pyqtSignal, pyqtSlot, pyqtProperty, Qt, QRegExp)
from PyQt5.QtWidgets import (QWidget, QFrame, QLineEdit, QLabel,
                             QScrollArea,
                             QHBoxLayout, QVBoxLayout)
from PyQt5.QtGui import (QDoubleValidator, QRegExpValidator)
from .QTrap import QTrap
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def getWidth():
    '''Get width of line edit in screen pixels'''
    edit = QLineEdit()
    fm = edit.fontMetrics()
    return fm.boundingRect('12345.6').width()


class QTrapPropertyEdit(QLineEdit):

    """Control for one property of one trap"""

    valueChanged = pyqtSignal(str, float)

    def __init__(self, name, value, decimals=2, *args, **kwargs):
        super(QTrapPropertyEdit, self).__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignRight)
        self.setFixedWidth(getWidth())
        self.setMaxLength(8)
        self.fmt = '{{:.{0}f}}'.format(decimals)
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

    @pyqtProperty(float)
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.setText(self.fmt.format(value))
        self._value = value


class QTrapListPropertyEdit(QLineEdit):

    """Control for one list-like property of one trap"""

    valueChanged = pyqtSignal(object, object)

    def __init__(self, name, value, *args, **kwargs):
        super(QTrapListPropertyEdit, self).__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignRight)
        self.setFixedWidth(getWidth())
        numberrx = '([+-]?\d*\.?\d+)'
        listrx = '\[' + '(?:\s*' + numberrx + '\s*,)*\s*' + numberrx + '\s*\]'
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

    @pyqtProperty(object)
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.setText(str(value))
        self._value = value


class QTrapPropertyWidget(QWidget):

    """Control for properties of one trap."""

    def __init__(self, trap, *args, **kwargs):
        super(QTrapPropertyWidget, self).__init__(*args, **kwargs)
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

    def __init__(self, *args, **kwargs):
        super(QTrapWidget, self).__init__(*args, **kwargs)
        self.properties = dict()
        self.init_ui()

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
            label.setFixedWidth(getWidth())
            layout.addWidget(label)
        widget.setLayout(layout)
        return widget

    @pyqtSlot(QTrap)
    def registerTrap(self, trap):
        trapWidget = QTrapPropertyWidget(trap)
        self.properties[trap] = trapWidget
        self.layout.addWidget(trapWidget)
        trap.destroyed.connect(lambda: self.unregisterTrap(trap))

    @pyqtSlot(QTrap)
    def unregisterTrap(self, trap):
        try:
            self.properties[trap].deleteLater()
        except Exception as ex:
            logger.warning('{}'.format(ex))

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
