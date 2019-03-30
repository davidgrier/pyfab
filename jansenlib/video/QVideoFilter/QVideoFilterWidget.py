# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QVideoFilterWidget.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QVideoFilterWidget(object):
    def setupUi(self, QVideoFilterWidget):
        QVideoFilterWidget.setObjectName("QVideoFilterWidget")
        QVideoFilterWidget.resize(126, 102)
        QVideoFilterWidget.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalLayout = QtWidgets.QVBoxLayout(QVideoFilterWidget)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.median = QtWidgets.QCheckBox(QVideoFilterWidget)
        self.median.setObjectName("median")
        self.verticalLayout.addWidget(self.median)
        self.deflicker = QtWidgets.QCheckBox(QVideoFilterWidget)
        self.deflicker.setObjectName("deflicker")
        self.verticalLayout.addWidget(self.deflicker)
        self.normalize = QtWidgets.QCheckBox(QVideoFilterWidget)
        self.normalize.setObjectName("normalize")
        self.verticalLayout.addWidget(self.normalize)
        self.samplehold = QtWidgets.QCheckBox(QVideoFilterWidget)
        self.samplehold.setObjectName("samplehold")
        self.verticalLayout.addWidget(self.samplehold)
        self.ndvi = QtWidgets.QCheckBox(QVideoFilterWidget)
        self.ndvi.setObjectName("ndvi")
        self.verticalLayout.addWidget(self.ndvi)
        self.detect = QtWidgets.QCheckBox(QVideoFilterWidget)
        self.detect.setObjectName("detect")
        self.verticalLayout.addWidget(self.detect)

        self.retranslateUi(QVideoFilterWidget)
        QtCore.QMetaObject.connectSlotsByName(QVideoFilterWidget)

    def retranslateUi(self, QVideoFilterWidget):
        _translate = QtCore.QCoreApplication.translate
        QVideoFilterWidget.setWindowTitle(_translate("QVideoFilterWidget", "Frame"))
        self.median.setText(_translate("QVideoFilterWidget", "Median"))
        self.deflicker.setText(_translate("QVideoFilterWidget", "Deflicker"))
        self.normalize.setText(_translate("QVideoFilterWidget", "Normalize"))
        self.samplehold.setText(_translate("QVideoFilterWidget", "Sample and Hold"))
        self.ndvi.setText(_translate("QVideoFilterWidget", "NDVI"))
        self.detect.setText(_translate("QVideoFilterWidget", "Detect"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QVideoFilterWidget = QtWidgets.QFrame()
    ui = Ui_QVideoFilterWidget()
    ui.setupUi(QVideoFilterWidget)
    QVideoFilterWidget.show()
    sys.exit(app.exec_())

