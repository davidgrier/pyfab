# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QOpenCVWidget.ui'
#
# Created by: PyQt5 UI code generator 5.5.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QOpenCVWidget(object):
    def setupUi(self, QOpenCVWidget):
        QOpenCVWidget.setObjectName("QOpenCVWidget")
        QOpenCVWidget.resize(212, 54)
        QOpenCVWidget.setMinimumSize(QtCore.QSize(212, 54))
        font = QtGui.QFont()
        font.setFamily("Arial")
        QOpenCVWidget.setFont(font)
        QOpenCVWidget.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalLayout = QtWidgets.QVBoxLayout(QOpenCVWidget)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frameProcessing = QtWidgets.QFrame(QOpenCVWidget)
        self.frameProcessing.setFrameShape(QtWidgets.QFrame.Panel)
        self.frameProcessing.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameProcessing.setObjectName("frameProcessing")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frameProcessing)
        self.horizontalLayout.setContentsMargins(3, 1, 3, 1)
        self.horizontalLayout.setSpacing(1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.mirrored = QtWidgets.QCheckBox(self.frameProcessing)
        self.mirrored.setObjectName("mirrored")
        self.horizontalLayout.addWidget(self.mirrored)
        self.flipped = QtWidgets.QCheckBox(self.frameProcessing)
        self.flipped.setObjectName("flipped")
        self.horizontalLayout.addWidget(self.flipped)
        self.gray = QtWidgets.QCheckBox(self.frameProcessing)
        self.gray.setObjectName("gray")
        self.horizontalLayout.addWidget(self.gray)
        self.verticalLayout.addWidget(self.frameProcessing)
        self.frameGeometry = QtWidgets.QFrame(QOpenCVWidget)
        self.frameGeometry.setFrameShape(QtWidgets.QFrame.Panel)
        self.frameGeometry.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameGeometry.setObjectName("frameGeometry")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frameGeometry)
        self.horizontalLayout_2.setContentsMargins(3, 1, 3, 1)
        self.horizontalLayout_2.setSpacing(1)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.labelWidth = QtWidgets.QLabel(self.frameGeometry)
        self.labelWidth.setObjectName("labelWidth")
        self.horizontalLayout_2.addWidget(self.labelWidth)
        self.width = QtWidgets.QSpinBox(self.frameGeometry)
        self.width.setObjectName("width")
        self.horizontalLayout_2.addWidget(self.width)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem)
        self.labelHeight = QtWidgets.QLabel(self.frameGeometry)
        self.labelHeight.setObjectName("labelHeight")
        self.horizontalLayout_2.addWidget(self.labelHeight)
        self.height = QtWidgets.QSpinBox(self.frameGeometry)
        self.height.setObjectName("height")
        self.horizontalLayout_2.addWidget(self.height)
        self.verticalLayout.addWidget(self.frameGeometry)
        self.labelWidth.setBuddy(self.width)
        self.labelHeight.setBuddy(self.height)

        self.retranslateUi(QOpenCVWidget)
        QtCore.QMetaObject.connectSlotsByName(QOpenCVWidget)

    def retranslateUi(self, QOpenCVWidget):
        _translate = QtCore.QCoreApplication.translate
        QOpenCVWidget.setWindowTitle(_translate("QOpenCVWidget", "Form"))
        self.mirrored.setStatusTip(_translate("QOpenCVWidget", "Camera: Flip about vertical axis"))
        self.mirrored.setText(_translate("QOpenCVWidget", "Mirrored"))
        self.flipped.setStatusTip(_translate("QOpenCVWidget", "Camera: Flip image about horizontal axis"))
        self.flipped.setText(_translate("QOpenCVWidget", "Flipped"))
        self.gray.setStatusTip(_translate("QOpenCVWidget", "Camera: Grayscale images"))
        self.gray.setText(_translate("QOpenCVWidget", "Gray"))
        self.labelWidth.setText(_translate("QOpenCVWidget", "Width"))
        self.width.setStatusTip(_translate("QOpenCVWidget", "Camera: Image width"))
        self.labelHeight.setText(_translate("QOpenCVWidget", "Height"))
        self.height.setStatusTip(_translate("QOpenCVWidget", "Camera: Image height"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QOpenCVWidget = QtWidgets.QFrame()
    ui = Ui_QOpenCVWidget()
    ui.setupUi(QOpenCVWidget)
    QOpenCVWidget.show()
    sys.exit(app.exec_())

