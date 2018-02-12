from PyQt4 import QtGui, QtCore
import numpy as np


class QNumericProperty(QtGui.QLineEdit):

    valueChanged = QtCore.pyqtSignal(object, object)

    def __init__(self, name, value, min, max):
        super(QNumericProperty, self).__init__()
        self.setAlignment(QtCore.Qt.AlignRight)
        self.name = name
        self.type = type(value)
        if self.type is int:
            v = QtGui.QIntValidator(int(min), int(max))
        elif self.type is float:
            v = QtGui.QDoubleValidator(float(min), float(max), 4)
            v.setNotation(QtGui.QDoubleValidator.StandardNotation)
        else:
            v = None
        self.setValidator(v)
        self.min = self.type(min)
        self.max = self.type(max)
        self.value = value
        self.returnPressed.connect(self.updateValue)

    @QtCore.pyqtSlot()
    def updateValue(self):
        self._value = self.type(str(self.text()))
        self.valueChanged.emit(self.name, self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, _value):
        value = np.clip(self.type(_value), self.min, self.max)
        self.setText(QtCore.QString('%.4f' % value))
        self.updateValue()


class QBooleanProperty(QtGui.QCheckBox):

    valueChanged = QtCore.pyqtSignal(str, bool)

    def __init__(self, name, value):
        super(QBooleanProperty, self).__init__()
        self.name = name
        self.value = bool(value)
        self.stateChanged.connect(self.updateValue)

    def updateValue(self, state):
        self._value = (state == QtCore.Qt.Checked)
        self.valueChanged.emit(self.name, self._value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        if bool(value):
            self.setCheckState(QtCore.Qt.Checked)
        else:
            self.setCheckState(QtCore.Qt.Unchecked)
        self.updateValue(self.checkState())


class QPropertySheet(QtGui.QFrame):

    def __init__(self, title=None, header=True, **kwargs):
        super(QPropertySheet, self).__init__(**kwargs)
        self.setFrameShape(QtGui.QFrame.Box)
        self.properties = dict()
        self.initUI(title, header)

    def initUI(self, title, header):
        self.layout = QtGui.QGridLayout(self)
        self.layout.setMargin(3)
        self.layout.setHorizontalSpacing(10)
        self.layout.setVerticalSpacing(3)
        self.setLayout(self.layout)
        self.row = 1
        if title is not None:
            self.layout.addWidget(QtGui.QLabel(title), self.row, 1, 1, 4)
            self.row += 1
        if header is True:
            self.layout.addWidget(QtGui.QLabel('property'), self.row, 1)
            label = QtGui.QLabel('value')
            label.setAlignment(QtCore.Qt.AlignCenter)
            self.layout.addWidget(label, self.row, 2)
            self.layout.addWidget(QtGui.QLabel('min'), self.row, 3)
            self.layout.addWidget(QtGui.QLabel('max'), self.row, 4)
            self.row += 1

    def registerProperty(self, name, value, min=None, max=None,
                         callback=None):
        wname = QtGui.QLabel(QtCore.QString(name))
        wname.setAlignment(QtCore.Qt.AlignRight)
        if isinstance(value, bool):
            wvalue = QBooleanProperty(name, value)
        else:
            wvalue = QNumericProperty(name, value, min, max)
        self.layout.addWidget(wname, self.row, 1)
        self.layout.addWidget(wvalue, self.row, 2)
        if min is not None:
            wmin = QtGui.QLabel(QtCore.QString(str(min)))
            wmax = QtGui.QLabel(QtCore.QString(str(max)))
            self.layout.addWidget(wmin, self.row, 3)
            self.layout.addWidget(wmax, self.row, 4)
        if callback is not None:
            wvalue.valueChanged.connect(callback)
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
