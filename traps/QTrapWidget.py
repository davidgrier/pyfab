from PyQt4.QtGui import QFrame, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea
from PyQt4.QtGui import QLineEdit, QDoubleValidator
from PyQt4.QtCore import Qt, pyqtSignal, QString


class QTrapProperty(QLineEdit):

    valueChanged = pyqtSignal(float)

    def __init__(self, value, decimals=1):
        super(QTrapProperty, self).__init__()
        self.setAlignment(Qt.AlignRight)
        self.setMaximumWidth(50)
        self.setMaxLength(7)
        self.fmt = '%' + '.%df' % decimals
        v = QDoubleValidator(decimals=decimals)
        v.setNotation(QDoubleValidator.StandardNotation)
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
        self.setText(QString(self.fmt % value))
        self.updateValue()


class QTrapLine(QWidget):

    def __init__(self, trap):
        super(QTrapLine, self).__init__()
        layout = QHBoxLayout()
        layout.setSpacing(1)
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


class QTrapWidget(QFrame):

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
        self.layout.setSpacing(1)
        self.layout.setMargin(1)
        self.layout.setAlignment(Qt.AlignTop)
        inner.setLayout(self.layout)
        scroll = QScrollArea()
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        self.setLayout(layout)

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
