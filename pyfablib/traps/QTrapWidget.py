# -*- coding: utf-8 -*-

"""Control panel for trap properties."""

from PyQt4 import QtGui, QtCore
try:
    from PyQt4.QtCore import QString
except ImportError:
    QString = str
from .QTrap import QTrap


class QTrapProperty(QtGui.QLineEdit):
    """Control for one property of one trap."""

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, value, decimals=1):
        super(QTrapProperty, self).__init__()
        self.setAlignment(QtCore.Qt.AlignRight)
        self.setMaximumWidth(60)
        self.setMaxLength(8)
        self.fmt = '%.{}f'.format(decimals)
        v = QtGui.QDoubleValidator(decimals=decimals)
        v.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.setValidator(v)
        self.value = value
        self.returnPressed.connect(self.updateValue)

    @QtCore.pyqtSlot()
    def updateValue(self):
        self.value = float(str(self.text()))
        self.valueChanged.emit(self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.setText(QString(self.fmt % value))
        self._value = value


class QTrapLine(QtGui.QWidget):
    """Control for properties of one trap."""

    def __init__(self, trap):
        super(QTrapLine, self).__init__()
        layout = QtGui.QHBoxLayout()
        layout.setSpacing(0)
        layout.setMargin(0)
        layout.setAlignment(QtCore.Qt.AlignLeft)
        self.wid = dict()
        for prop in trap.properties:
            name = prop['name']
            self.wid[name] = self.propertyWidget(trap, prop)
            tip = trap.__class__.__name__ + ': ' + name
            self.wid[name].setStatusTip(tip)
            layout.addWidget(self.wid[name])
        trap.valueChanged.connect(self.updateValues)
        self.setLayout(layout)

    def propertyWidget(self, trap, prop):
        name = prop['name']
        decimals = prop['decimals']
        value = getattr(trap, name)
        wid = QTrapProperty(value, decimals=decimals)
        wid.valueChanged.connect(lambda v: trap.setProperty(name, v))
        return wid

    @QtCore.pyqtSlot(QTrap)
    def updateValues(self, trap):
        for prop in trap.properties:
            name = prop['name']
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
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout = QtGui.QVBoxLayout(self)
        layout.addWidget(scroll)
        self.setLayout(layout)
        self.layout.addWidget(self.labelLine())

    def labelItem(self, name):
        label = QtGui.QLabel(name)
        label.setAlignment(QtCore.Qt.AlignCenter)
        label.setMinimumWidth(60)
        return label

    def labelLine(self):
        widget = QtGui.QWidget()
        layout = QtGui.QHBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignLeft)
        layout.setSpacing(0)
        layout.setMargin(0)
        for label in ['x', 'y', 'z', 'alpha', 'phi']:
            layout.addWidget(self.labelItem(label))
        widget.setLayout(layout)
        return widget

    def registerTrap(self, trap):
        trapline = QTrapLine(trap)
        self.properties[trap] = trapline
        self.layout.addWidget(trapline)
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
