# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tasks/DefaultTaskWidget.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_DefaultTaskWidget(object):
    def setupUi(self, DefaultTaskWidget):
        DefaultTaskWidget.setObjectName("DefaultTaskWidget")
        DefaultTaskWidget.resize(275, 449)
        self.gridLayout_2 = QtWidgets.QGridLayout(DefaultTaskWidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.widget_3 = QtWidgets.QWidget(DefaultTaskWidget)
        self.widget_3.setMaximumSize(QtCore.QSize(16777215, 100))
        self.widget_3.setObjectName("widget_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget_3)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self._initialized = QtWidgets.QCheckBox(self.widget_3)
        self._initialized.setEnabled(False)
        self._initialized.setObjectName("_initialized")
        self.verticalLayout.addWidget(self._initialized)
        self._blocking = QtWidgets.QCheckBox(self.widget_3)
        self._blocking.setEnabled(False)
        self._blocking.setChecked(False)
        self._blocking.setObjectName("_blocking")
        self.verticalLayout.addWidget(self._blocking)
        self._paused = QtWidgets.QCheckBox(self.widget_3)
        self._paused.setEnabled(False)
        self._paused.setObjectName("_paused")
        self.verticalLayout.addWidget(self._paused)
        self._busy = QtWidgets.QCheckBox(self.widget_3)
        self._busy.setEnabled(False)
        self._busy.setObjectName("_busy")
        self.verticalLayout.addWidget(self._busy)
        self.gridLayout_2.addWidget(self.widget_3, 1, 0, 1, 2)
        self.name = QtWidgets.QLabel(DefaultTaskWidget)
        self.name.setMaximumSize(QtCore.QSize(16777215, 25))
        self.name.setObjectName("name")
        self.gridLayout_2.addWidget(self.name, 0, 0, 1, 1)
        self.widget_2 = QtWidgets.QWidget(DefaultTaskWidget)
        self.widget_2.setMaximumSize(QtCore.QSize(16777215, 100))
        self.widget_2.setObjectName("widget_2")
        self.gridLayout = QtWidgets.QGridLayout(self.widget_2)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.skip = QtWidgets.QSpinBox(self.widget_2)
        self.skip.setMinimum(1)
        self.skip.setObjectName("skip")
        self.gridLayout.addWidget(self.skip, 1, 1, 1, 1)
        self.lskip = QtWidgets.QLabel(self.widget_2)
        self.lskip.setObjectName("lskip")
        self.gridLayout.addWidget(self.lskip, 1, 0, 1, 1)
        self.ldelay = QtWidgets.QLabel(self.widget_2)
        self.ldelay.setObjectName("ldelay")
        self.gridLayout.addWidget(self.ldelay, 0, 0, 1, 1)
        self.lframe = QtWidgets.QLabel(self.widget_2)
        self.lframe.setObjectName("lframe")
        self.gridLayout.addWidget(self.lframe, 2, 0, 1, 1)
        self.delay = QtWidgets.QSpinBox(self.widget_2)
        self.delay.setObjectName("delay")
        self.gridLayout.addWidget(self.delay, 0, 1, 1, 1)
        self._frame = QtWidgets.QLabel(self.widget_2)
        self._frame.setText("")
        self._frame.setObjectName("_frame")
        self.gridLayout.addWidget(self._frame, 2, 1, 1, 1)
        self.lnframes = QtWidgets.QLabel(self.widget_2)
        self.lnframes.setObjectName("lnframes")
        self.gridLayout.addWidget(self.lnframes, 3, 0, 1, 1)
        self.nframes = QtWidgets.QSpinBox(self.widget_2)
        self.nframes.setMaximum(1000)
        self.nframes.setObjectName("nframes")
        self.gridLayout.addWidget(self.nframes, 3, 1, 1, 1)
        self.gridLayout_2.addWidget(self.widget_2, 1, 2, 1, 1)
        self.settingsView = QtWidgets.QWidget(DefaultTaskWidget)
        self.settingsView.setObjectName("settingsView")
        self.gridLayout_2.addWidget(self.settingsView, 2, 0, 1, 3)

        self.retranslateUi(DefaultTaskWidget)
        QtCore.QMetaObject.connectSlotsByName(DefaultTaskWidget)

    def retranslateUi(self, DefaultTaskWidget):
        _translate = QtCore.QCoreApplication.translate
        DefaultTaskWidget.setWindowTitle(_translate("DefaultTaskWidget", "Form"))
        self._initialized.setText(_translate("DefaultTaskWidget", "initialized"))
        self._blocking.setText(_translate("DefaultTaskWidget", "blocking"))
        self._paused.setText(_translate("DefaultTaskWidget", "paused"))
        self._busy.setText(_translate("DefaultTaskWidget", "busy"))
        self.name.setText(_translate("DefaultTaskWidget", "taskname"))
        self.lskip.setText(_translate("DefaultTaskWidget", "skip"))
        self.ldelay.setText(_translate("DefaultTaskWidget", "delay"))
        self.lframe.setText(_translate("DefaultTaskWidget", "frame"))
        self.lnframes.setText(_translate("DefaultTaskWidget", "nframes"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    DefaultTaskWidget = QtWidgets.QWidget()
    ui = Ui_DefaultTaskWidget()
    ui.setupUi(DefaultTaskWidget)
    DefaultTaskWidget.show()
    sys.exit(app.exec_())

