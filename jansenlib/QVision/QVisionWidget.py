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
        QVisionWidget.setEnabled(True)
        QVisionWidget.resize(321, 592)
        self.groupExport = QtWidgets.QGroupBox(QVisionWidget)
        self.groupExport.setGeometry(QtCore.QRect(20, 130, 281, 81))
        self.groupExport.setObjectName("groupExport")
        self.checkFrames = QtWidgets.QCheckBox(self.groupExport)
        self.checkFrames.setGeometry(QtCore.QRect(10, 20, 151, 21))
        self.checkFrames.setChecked(False)
        self.checkFrames.setTristate(False)
        self.checkFrames.setObjectName("checkFrames")
        self.checkDiscard = QtWidgets.QCheckBox(self.groupExport)
        self.checkDiscard.setGeometry(QtCore.QRect(30, 40, 141, 21))
        self.checkDiscard.setObjectName("checkDiscard")
        self.checkTrajectories = QtWidgets.QCheckBox(self.groupExport)
        self.checkTrajectories.setGeometry(QtCore.QRect(10, 60, 171, 21))
        self.checkTrajectories.setObjectName("checkTrajectories")
        self.groupPipeline = QtWidgets.QGroupBox(QVisionWidget)
        self.groupPipeline.setGeometry(QtCore.QRect(20, 90, 281, 41))
        self.groupPipeline.setObjectName("groupPipeline")
        self.bEstimate = QtWidgets.QRadioButton(self.groupPipeline)
        self.bEstimate.setGeometry(QtCore.QRect(90, 20, 91, 21))
        self.bEstimate.setObjectName("bEstimate")
        self.bRefine = QtWidgets.QRadioButton(self.groupPipeline)
        self.bRefine.setGeometry(QtCore.QRect(180, 20, 81, 21))
        self.bRefine.setObjectName("bRefine")
        self.bDetect = QtWidgets.QRadioButton(self.groupPipeline)
        self.bDetect.setGeometry(QtCore.QRect(10, 20, 71, 21))
        self.bDetect.setObjectName("bDetect")
        self.groupProcess = QtWidgets.QGroupBox(QVisionWidget)
        self.groupProcess.setGeometry(QtCore.QRect(20, 9, 281, 81))
        self.groupProcess.setObjectName("groupProcess")
        self.breal = QtWidgets.QRadioButton(self.groupProcess)
        self.breal.setGeometry(QtCore.QRect(10, 20, 121, 21))
        self.breal.setObjectName("breal")
        self.bpost = QtWidgets.QRadioButton(self.groupProcess)
        self.bpost.setGeometry(QtCore.QRect(10, 60, 131, 21))
        self.bpost.setChecked(True)
        self.bpost.setObjectName("bpost")
        self.label = QtWidgets.QLabel(self.groupProcess)
        self.label.setGeometry(QtCore.QRect(30, 40, 41, 21))
        self.label.setObjectName("label")
        self.skipBox = QtWidgets.QSpinBox(self.groupProcess)
        self.skipBox.setGeometry(QtCore.QRect(70, 40, 91, 22))
        self.skipBox.setPrefix("")
        self.skipBox.setMaximum(50)
        self.skipBox.setProperty("value", 5)
        self.skipBox.setObjectName("skipBox")
        self.plot = PlotWidget(QVisionWidget)
        self.plot.setEnabled(True)
        self.plot.setGeometry(QtCore.QRect(30, 220, 261, 211))
        self.plot.setObjectName("plot")

        self.retranslateUi(QVisionWidget)
        QtCore.QMetaObject.connectSlotsByName(QVisionWidget)

    def retranslateUi(self, QVisionWidget):
        _translate = QtCore.QCoreApplication.translate
        QVisionWidget.setWindowTitle(_translate("QVisionWidget", "Form"))
        self.groupExport.setTitle(_translate("QVisionWidget", "Export options"))
        self.checkFrames.setText(_translate("QVisionWidget", "Save frames"))
        self.checkDiscard.setText(_translate("QVisionWidget", "Discard empty"))
        self.checkTrajectories.setText(_translate("QVisionWidget", "Save trajectories"))
        self.groupPipeline.setTitle(_translate("QVisionWidget", "Vision pipeline"))
        self.bEstimate.setText(_translate("QVisionWidget", "Estimate"))
        self.bRefine.setText(_translate("QVisionWidget", "Refine"))
        self.bDetect.setText(_translate("QVisionWidget", "Detect"))
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

