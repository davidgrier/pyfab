from PyQt4.QtGui import QFrame, QGridLayout, QLabel, QLineEdit, QCheckBox
from PyQt4.QtGui import QIntValidator, QDoubleValidator
from PyQt4.QtCore import Qt, QString, pyqtSignal
import numpy as np


class QFabProperty(QLineEdit):

    valueChanged = pyqtSignal()

    def __init__(self, name, value, min, max):
        super(QFabProperty, self).__init__()
        self.setAlignment(Qt.AlignRight)
        self.type = type(value)
        if self.type is int:
            v = QIntValidator(int(min), int(max))
        elif self.type is float:
            v = QDoubleValidator(float(min), float(max), 4)
            v.setNotation(QDoubleValidator.StandardNotation)
        else:
            v = None
        self.setValidator(v)
        self.min = self.type(min)
        self.max = self.type(max)
        self.value = value
        self.returnPressed.connect(self.updateValue)

    def updateValue(self):
        self._value = self.type(str(self.text()))
        self.valueChanged.emit()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, _value):
        value = np.clip(self.type(_value), self.min, self.max)
        self.setText(QString('%.4f' % value))
        self.updateValue()


class QFabBoolean(QCheckBox):

    valueChanged = pyqtSignal()

    def __init__(self, name, value):
        super(QFabBoolean, self).__init__()
        self.value = bool(value)
        self.stateChanged.connect(self.updateValue)

    def updateValue(self, state):
        self._value = (state == Qt.Checked)
        self.valueChanged.emit()

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if bool(value):
            self.setCheckState(Qt.Checked)
        else:
            self.setCheckState(Qt.Unchecked)
        self.updateValue(self.checkState())


class QPropertySheet(QFrame):

    def __init__(self, title=None, header=True, **kwargs):
        super(QPropertySheet, self).__init__(**kwargs)
        self.setFrameShape(QFrame.Box)
        self.properties = dict()
        self.initUI(title, header)

    def initUI(self, title, header):
        self.layout = QGridLayout(self)
        self.layout.setMargin(3)
        self.layout.setHorizontalSpacing(10)
        self.layout.setVerticalSpacing(3)
        self.setLayout(self.layout)
        self.row = 1
        if title is not None:
            self.layout.addWidget(QLabel(title), self.row, 1, 1, 4)
            self.row += 1
        if header is True:
            self.layout.addWidget(QLabel('property'), self.row, 1)
            label = QLabel('value')
            label.setAlignment(Qt.AlignCenter)
            self.layout.addWidget(label, self.row, 2)
            self.layout.addWidget(QLabel('min'), self.row, 3)
            self.layout.addWidget(QLabel('max'), self.row, 4)
            self.row += 1

    def registerProperty(self, name, value, min=None, max=None):
        wname = QLabel(QString(name))
        wname.setAlignment(Qt.AlignRight)
        if isinstance(value, bool):
            wvalue = QFabBoolean(name, value)
        else:
            wvalue = QFabProperty(name, value, min, max)
        self.layout.addWidget(wname, self.row, 1)
        self.layout.addWidget(wvalue, self.row, 2)
        if min is not None:
            wmin = QLabel(QString(str(min)))
            wmax = QLabel(QString(str(max)))
            self.layout.addWidget(wmin, self.row, 3)
            self.layout.addWidget(wmax, self.row, 4)
        self.row += 1
        self.properties[name] = wvalue
        return wvalue

    @property
    def enabled(self):
        result = True
        for propname in self.properties:
            prop = self.properties[propname]
            result = result and prop.isEnabled()
        return(result)

    @enabled.setter
    def enabled(self, value):
        state = bool(value)
        for propname in self.properties:
            prop = self.properties[propname]
            prop.setEnabled(state)


def main():
    import sys
    from PyQt4.QtGui import QApplication

    app = QApplication(sys.argv)
    sheet = QPropertySheet()
    wxc = sheet.registerProperty('xc', 10, -100, 100)
    wyc = sheet.registerProperty('yc', 200, -100, 100)
    walpha = sheet.registerProperty('alpha', 5., 0., 10.)
    wbool = sheet.registerProperty('bool', True)
    sheet.show()
    print(wxc.value, wyc.value, walpha.value, wbool.value)
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
