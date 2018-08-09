# -*- coding: utf-8 -*-

"""Control panel for trap properties."""

from PyQt4 import QtGui, QtCore
try:
    from PyQt4.QtCore import QString
except ImportError:
    QString = str
from .QTrap import QTrap
import numpy as np


class QTrapPropertyEdit(QtGui.QLineEdit):
    """Control for one property of one trap"""

    valueChanged = QtCore.pyqtSignal(object, float)

    def __init__(self, name, value, decimals=1):
        super(QTrapPropertyEdit, self).__init__()
        self.setAlignment(QtCore.Qt.AlignRight)
        self.setFixedWidth(50)
        self.setMaxLength(8)
        self.fmt = '%.{}f'.format(decimals)
        v = QtGui.QDoubleValidator(decimals=decimals)
        v.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.setValidator(v)
        self.name = name
        self.value = value
        self.returnPressed.connect(self.updateValue)

    @QtCore.pyqtSlot()
    def updateValue(self):
        self.value = float(str(self.text()))
        self.valueChanged.emit(self.name, self.value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.setText(QString(self.fmt % value))
        self._value = value


class QTrapListPropertyEdit(QtGui.QLineEdit):
    """Control for one list-like property of one trap"""

    valueChanged = QtCore.pyqtSignal(object, object)

    def __init__(self, name, value):
        super(QTrapListPropertyEdit, self).__init__()
        self.setAlignment(QtCore.Qt.AlignRight)
        self.setFixedWidth(50)
        numberrx = '([+-]?\d+\.?\d*)'
        listrx = '\[' + '(?:\s*'+numberrx+'\s*,)*\s*' + numberrx + '\s*\]'
        print(listrx)
        self.rx = QtCore.QRegExp(listrx)
        val = QtGui.QRegExpValidator(self.rx)
        self.setValidator(val)
        self.name = name
        self.value = value
        self.returnPressed.connect(self.updateValue)

    @QtCore.pyqtSlot()
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


class QTrapPropertyWidget(QtGui.QWidget):
    """Control for properties of one trap."""

    def __init__(self, trap):
        super(QTrapPropertyWidget, self).__init__()
        layout = QtGui.QHBoxLayout()
        layout.setSpacing(0)
        layout.setMargin(0)
        layout.setAlignment(QtCore.Qt.AlignLeft)
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

    @QtCore.pyqtSlot(QTrap)
    def updateValues(self, trap):
        for name in trap.properties.keys():
            value = getattr(trap, name)
            self.wid[name].value = value


class QTrapWidget(QtGui.QFrame):
    """Controls for all traps."""

    def __init__(self, pattern=None):
        super(QTrapWidget, self).__init__()
        self.properties = dict()
        self.init_ui()
        if pattern is not None:
            pattern.trapAdded.connect(self.registerTrap)

    def init_ui(self):
        self.setFrameShape(QtGui.QFrame.Box)
        inner = QtGui.QWidget()
        self.layout = QtGui.QVBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setMargin(0)
        self.layout.setAlignment(QtCore.Qt.AlignTop)
        inner.setLayout(self.layout)
        scroll = QtGui.QScrollArea()
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.layout.addWidget(self.labelLine())

    def labelLine(self):
        widget = QtGui.QWidget()
        layout = QtGui.QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignLeft)
        layout.setSpacing(0)
        layout.setMargin(0)
        for name in ['x', 'y', 'z', 'alpha', 'phi']:
            label = QtGui.QLabel(name)
            label.setAlignment(QtCore.Qt.AlignCenter)
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
    from PyQt4 import QtGui

    app = QtGui.QApplication(sys.argv)
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
