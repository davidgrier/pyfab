# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QVisionWidget.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QVisionWidget(object):
    def setupUi(self, QVisionWidget):
        QVisionWidget.setObjectName("QVisionWidget")
        QVisionWidget.resize(400, 314)
        self.groupExport = QtWidgets.QGroupBox(QVisionWidget)
        self.groupExport.setGeometry(QtCore.QRect(10, 170, 382, 121))
        self.groupExport.setObjectName("groupExport")
        self.checkFrames = QtWidgets.QCheckBox(self.groupExport)
        self.checkFrames.setGeometry(QtCore.QRect(10, 60, 271, 16))
        self.checkFrames.setChecked(True)
        self.checkFrames.setTristate(False)
        self.checkFrames.setObjectName("checkFrames")
        self.checkTrajectories = QtWidgets.QCheckBox(self.groupExport)
        self.checkTrajectories.setGeometry(QtCore.QRect(10, 100, 267, 21))
        self.checkTrajectories.setObjectName("checkTrajectories")
        self.lineSave = QtWidgets.QLineEdit(self.groupExport)
        self.lineSave.setGeometry(QtCore.QRect(60, 30, 211, 21))
        self.lineSave.setObjectName("lineSave")
        self.labelSave = QtWidgets.QLabel(self.groupExport)
        self.labelSave.setGeometry(QtCore.QRect(10, 30, 51, 21))
        self.labelSave.setObjectName("labelSave")
        self.bdiscard = QtWidgets.QRadioButton(self.groupExport)
        self.bdiscard.setGeometry(QtCore.QRect(40, 80, 111, 16))
        self.bdiscard.setObjectName("bdiscard")
        self.groupPipeline = QtWidgets.QGroupBox(QVisionWidget)
        self.groupPipeline.setGeometry(QtCore.QRect(9, 105, 381, 63))
        self.groupPipeline.setObjectName("groupPipeline")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.groupPipeline)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.checkDetect = QtWidgets.QCheckBox(self.groupPipeline)
        self.checkDetect.setObjectName("checkDetect")
        self.horizontalLayout_5.addWidget(self.checkDetect)
        self.checkEstimate = QtWidgets.QCheckBox(self.groupPipeline)
        self.checkEstimate.setObjectName("checkEstimate")
        self.horizontalLayout_5.addWidget(self.checkEstimate)
        self.checkRefine = QtWidgets.QCheckBox(self.groupPipeline)
        self.checkRefine.setObjectName("checkRefine")
        self.horizontalLayout_5.addWidget(self.checkRefine)
        self.groupProcess = QtWidgets.QGroupBox(QVisionWidget)
        self.groupProcess.setGeometry(QtCore.QRect(9, 9, 382, 90))
        self.groupProcess.setObjectName("groupProcess")
        self.label = QtWidgets.QLabel(self.groupProcess)
        self.label.setGeometry(QtCore.QRect(10, 60, 31, 21))
        self.label.setObjectName("label")
        self.bpost = QtWidgets.QRadioButton(self.groupProcess)
        self.bpost.setGeometry(QtCore.QRect(12, 30, 83, 21))
        self.bpost.setObjectName("bpost")
        self.breal = QtWidgets.QRadioButton(self.groupProcess)
        self.breal.setGeometry(QtCore.QRect(100, 30, 102, 21))
        self.breal.setChecked(True)
        self.breal.setObjectName("breal")
        self.skipBox = QtWidgets.QSpinBox(self.groupProcess)
        self.skipBox.setGeometry(QtCore.QRect(40, 60, 91, 21))
        self.skipBox.setPrefix("")
        self.skipBox.setMaximum(50)
        self.skipBox.setProperty("value", 5)
        self.skipBox.setObjectName("skipBox")

        self.retranslateUi(QVisionWidget)
        QtCore.QMetaObject.connectSlotsByName(QVisionWidget)

    def retranslateUi(self, QVisionWidget):
        _translate = QtCore.QCoreApplication.translate
        QVisionWidget.setWindowTitle(_translate("QVisionWidget", "Form"))
        self.groupExport.setTitle(_translate("QVisionWidget", "Export options"))
        self.checkFrames.setText(_translate("QVisionWidget", "Save frames"))
        self.checkTrajectories.setText(_translate("QVisionWidget", "Save trajectories"))
        self.labelSave.setText(_translate("QVisionWidget", "Save as"))
        self.bdiscard.setText(_translate("QVisionWidget", "Discard empty"))
        self.groupPipeline.setTitle(_translate("QVisionWidget", "Vision pipeline"))
        self.checkDetect.setText(_translate("QVisionWidget", "Detect"))
        self.checkEstimate.setText(_translate("QVisionWidget", "Estimate"))
        self.checkRefine.setText(_translate("QVisionWidget", "Refine"))
        self.groupProcess.setTitle(_translate("QVisionWidget", "Processing options"))
        self.label.setText(_translate("QVisionWidget", "Skip"))
        self.bpost.setText(_translate("QVisionWidget", "Real-time"))
        self.breal.setText(_translate("QVisionWidget", "Post-process"))
        self.skipBox.setSuffix(_translate("QVisionWidget", " frames"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QVisionWidget = QtWidgets.QWidget()
    ui = Ui_QVisionWidget()
    ui.setupUi(QVisionWidget)
    QVisionWidget.show()
    sys.exit(app.exec_())

