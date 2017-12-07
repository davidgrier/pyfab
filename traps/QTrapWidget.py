from PyQt4.QtGui import QFrame, QVBoxLayout, QHBoxLayout
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


class QTrapLine(QHBoxLayout):

    def __init__(self, trap):
        super(QTrapLine, self).__init__()
        self.setSpacing(1)
        self.setMargin(0)
        self.wx = QTrapProperty(trap.r.x())
        self.wy = QTrapProperty(trap.r.y())
        self.wz = QTrapProperty(trap.r.z())
        self.wa = QTrapProperty(trap.a, decimals=2)
        self.wp = QTrapProperty(trap.phi, decimals=2)
        self.addWidget(self.wx)
        self.addWidget(self.wy)
        self.addWidget(self.wz)
        self.addWidget(self.wa)
        self.addWidget(self.wp)
        trap.valueChanged.connect(self.updateValues)
        self.wx.valueChanged.connect(trap.r.setX)
        self.wy.valueChanged.connect(trap.r.setY)
        self.wz.valueChanged.connect(trap.r.setZ)
        self.wa.valueChanged.connect(trap.setA)
        self.wp.valueChanged.connect(trap.setPhi)

    def updateValues(self, trap):
        self.wx.value = trap.r.x()
        self.wy.value = trap.r.y()
        self.wz.value = trap.r.z()
        self.wa.value = trap.a
        self.wp.value = trap.phi


class QTrapWidget(QFrame):

    def __init__(self, pattern=None):
        super(QTrapWidget, self).__init__()
        self.setFrameShape(QFrame.Box)
        self.properties = dict()
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(1)
        self.layout.setMargin(1)

    def registerTrap(self, trap):
        trapline = QTrapLine(trap)
        self.layout.addLayout(trapline)


if __name__ == '__main__':
    import sys
    from QTrap import QTrap
    from PyQt4 import QtGui

    app = QtGui.QApplication(sys.argv)
    wtrap = QTrapWidget()
    trapa = QTrap()
    trapb = QTrap()
    wtrap.registerTrap(trapa)
    wtrap.registerTrap(trapb)
    wtrap.show()
    trapa.r = (100, 100, 10)
    sys.exit(app.exec_())
