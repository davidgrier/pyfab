# -*- coding: utf-8 -*-

"""Control panel for trap properties."""

from PyQt4 import QtGui, QtCore


class QTrapProperty(QtGui.QLineEdit):
    """Control for one property of one trap."""

    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, value, decimals=1):
        super(QTrapProperty, self).__init__()
        self.setAlignment(QtCore.Qt.AlignRight)
        self.setMaximumWidth(50)
        self.setMaxLength(8)
        self.fmt = '%' + '.%df' % decimals
        v = QtGui.QDoubleValidator(decimals=decimals)
        v.setNotation(QtGui.QDoubleValidator.StandardNotation)
        self.setValidator(v)
        self.value = value
        self.returnPressed.connect(self.updateValue)

    def updateValue(self):
        self._value = float(str(self.text()))
        self.valueChanged.emit(self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self.setText(QtCore.QString(self.fmt % value))
        self.updateValue()


class QTrapLine(QtGui.QWidget):
    """Control for properties of one trap."""
    
    def __init__(self, trap):
        super(QTrapLine, self).__init__()
        layout = QtGui.QHBoxLayout()
        layout.setSpacing(0)
        layout.setMargin(0)
        self.wx = QTrapProperty(trap.r.x())
        self.wy = QTrapProperty(trap.r.y())
        self.wz = QTrapProperty(trap.r.z())
        self.wa = QTrapProperty(trap.a, decimals=2)
        self.wp = QTrapProperty(trap.phi, decimals=2)
        layout.addWidget(self.wx)
        layout.addWidget(self.wy)
        layout.addWidget(self.wz)
        layout.addWidget(self.wa)
        layout.addWidget(self.wp)
        trap.valueChanged.connect(self.updateValues)
        self.wx.valueChanged.connect(trap.setX)
        self.wy.valueChanged.connect(trap.setY)
        self.wz.valueChanged.connect(trap.setZ)
        self.wa.valueChanged.connect(trap.setA)
        self.wp.valueChanged.connect(trap.setPhi)
        self.setLayout(layout)

    def updateValues(self, trap):
        self.wx.value = trap.r.x()
        self.wy.value = trap.r.y()
        self.wz.value = trap.r.z()
        self.wa.value = trap.a
        self.wp.value = trap.phi


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
        label.setMaximumWidth(50)
        return label

    def labelLine(self):
        widget = QtGui.QWidget()
        layout = QtGui.QHBoxLayout()
        layout.setSpacing(0)
        layout.setMargin(0)
        layout.addWidget(self.labelItem('x'))
        layout.addWidget(self.labelItem('y'))
        layout.addWidget(self.labelItem('z'))
        layout.addWidget(self.labelItem('alpha'))
        layout.addWidget(self.labelItem('phi'))
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
