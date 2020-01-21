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
        QVisionWidget.resize(314, 519)
        self.groupExport = QtWidgets.QGroupBox(QVisionWidget)
        self.groupExport.setGeometry(QtCore.QRect(10, 170, 270, 90))
        self.groupExport.setObjectName("groupExport")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupExport)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.checkFrames = QtWidgets.QCheckBox(self.groupExport)
        self.checkFrames.setChecked(False)
        self.checkFrames.setTristate(False)
        self.checkFrames.setObjectName("checkFrames")
        self.gridLayout_2.addWidget(self.checkFrames, 0, 0, 1, 1)
        self.bdiscard = QtWidgets.QRadioButton(self.groupExport)
        self.bdiscard.setObjectName("bdiscard")
        self.gridLayout_2.addWidget(self.bdiscard, 0, 1, 1, 1)
        self.checkTrajectories = QtWidgets.QCheckBox(self.groupExport)
        self.checkTrajectories.setObjectName("checkTrajectories")
        self.gridLayout_2.addWidget(self.checkTrajectories, 1, 0, 1, 1)
        self.groupPipeline = QtWidgets.QGroupBox(QVisionWidget)
        self.groupPipeline.setGeometry(QtCore.QRect(9, 105, 271, 63))
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
        self.groupProcess.setGeometry(QtCore.QRect(9, 9, 211, 93))
        self.groupProcess.setObjectName("groupProcess")
        self.gridLayout = QtWidgets.QGridLayout(self.groupProcess)
        self.gridLayout.setObjectName("gridLayout")
        self.breal = QtWidgets.QRadioButton(self.groupProcess)
        self.breal.setObjectName("breal")
        self.gridLayout.addWidget(self.breal, 0, 0, 1, 2)
        self.bpost = QtWidgets.QRadioButton(self.groupProcess)
        self.bpost.setChecked(True)
        self.bpost.setObjectName("bpost")
        self.gridLayout.addWidget(self.bpost, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.groupProcess)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 1, 0, 1, 1)
        self.skipBox = QtWidgets.QSpinBox(self.groupProcess)
        self.skipBox.setPrefix("")
        self.skipBox.setMaximum(50)
        self.skipBox.setProperty("value", 5)
        self.skipBox.setObjectName("skipBox")
        self.gridLayout.addWidget(self.skipBox, 1, 1, 1, 2)
        self.plot = PlotWidget(QVisionWidget)
        self.plot.setGeometry(QtCore.QRect(20, 270, 261, 211))
        self.plot.setObjectName("plot")

        self.retranslateUi(QVisionWidget)
        QtCore.QMetaObject.connectSlotsByName(QVisionWidget)

    def retranslateUi(self, QVisionWidget):
        _translate = QtCore.QCoreApplication.translate
        QVisionWidget.setWindowTitle(_translate("QVisionWidget", "Form"))
        self.groupExport.setTitle(_translate("QVisionWidget", "Export options"))
        self.checkFrames.setText(_translate("QVisionWidget", "Save frames"))
        self.bdiscard.setText(_translate("QVisionWidget", "Discard empty"))
        self.checkTrajectories.setText(_translate("QVisionWidget", "Save trajectories"))
        self.groupPipeline.setTitle(_translate("QVisionWidget", "Vision pipeline"))
        self.checkDetect.setText(_translate("QVisionWidget", "Detect"))
        self.checkEstimate.setText(_translate("QVisionWidget", "Estimate"))
        self.checkRefine.setText(_translate("QVisionWidget", "Refine"))
        self.groupProcess.setTitle(_translate("QVisionWidget", "Processing options"))
        self.breal.setText(_translate("QVisionWidget", "Real-time"))
        self.bpost.setText(_translate("QVisionWidget", "Post-process"))
        self.label.setText(_translate("QVisionWidget", "Skip"))
        self.skipBox.setSuffix(_translate("QVisionWidget", " frames"))

from pyqtgraph import PlotWidget

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QVisionWidget = QtWidgets.QWidget()
    ui = Ui_QVisionWidget()
    ui.setupUi(QVisionWidget)
    QVisionWidget.show()
    sys.exit(app.exec_())

