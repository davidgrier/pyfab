# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QVisionWidget.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QVisionWidget(object):
    def setupUi(self, QVisionWidget):
        QVisionWidget.setObjectName("QVisionWidget")
        QVisionWidget.setEnabled(True)
        QVisionWidget.resize(550, 742)
        self.verticalLayout = QtWidgets.QVBoxLayout(QVisionWidget)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupProcess = QtWidgets.QGroupBox(QVisionWidget)
        self.groupProcess.setObjectName("groupProcess")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupProcess)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.breal = QtWidgets.QRadioButton(self.groupProcess)
        self.breal.setObjectName("breal")
        self.verticalLayout_3.addWidget(self.breal)
        self.bpost = QtWidgets.QRadioButton(self.groupProcess)
        self.bpost.setChecked(True)
        self.bpost.setObjectName("bpost")
        self.verticalLayout_3.addWidget(self.bpost)
        self.skipBox = QtWidgets.QSpinBox(self.groupProcess)
        self.skipBox.setMaximum(50)
        self.skipBox.setProperty("value", 5)
        self.skipBox.setObjectName("skipBox")
        self.verticalLayout_3.addWidget(self.skipBox)
        self.verticalLayout.addWidget(self.groupProcess)
        self.groupPipeline = QtWidgets.QGroupBox(QVisionWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupPipeline.sizePolicy().hasHeightForWidth())
        self.groupPipeline.setSizePolicy(sizePolicy)
        self.groupPipeline.setObjectName("groupPipeline")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupPipeline)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bDetect = QtWidgets.QCheckBox(self.groupPipeline)
        self.bDetect.setObjectName("bDetect")
        self.horizontalLayout.addWidget(self.bDetect)
        self.bEstimate = QtWidgets.QCheckBox(self.groupPipeline)
        self.bEstimate.setObjectName("bEstimate")
        self.horizontalLayout.addWidget(self.bEstimate)
        self.bRefine = QtWidgets.QCheckBox(self.groupPipeline)
        self.bRefine.setObjectName("bRefine")
        self.horizontalLayout.addWidget(self.bRefine)
        self.verticalLayout.addWidget(self.groupPipeline)
        self.groupExport = QtWidgets.QGroupBox(QVisionWidget)
        self.groupExport.setObjectName("groupExport")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupExport)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.checkFrames = QtWidgets.QCheckBox(self.groupExport)
        self.checkFrames.setChecked(False)
        self.checkFrames.setTristate(False)
        self.checkFrames.setObjectName("checkFrames")
        self.verticalLayout_2.addWidget(self.checkFrames)
        self.checkTrajectories = QtWidgets.QCheckBox(self.groupExport)
        self.checkTrajectories.setObjectName("checkTrajectories")
        self.verticalLayout_2.addWidget(self.checkTrajectories)
        self.verticalLayout.addWidget(self.groupExport)
        self.plot = PlotWidget(QVisionWidget)
        self.plot.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(2)
        sizePolicy.setHeightForWidth(self.plot.sizePolicy().hasHeightForWidth())
        self.plot.setSizePolicy(sizePolicy)
        self.plot.setObjectName("plot")
        self.verticalLayout.addWidget(self.plot)

        self.retranslateUi(QVisionWidget)
        QtCore.QMetaObject.connectSlotsByName(QVisionWidget)

    def retranslateUi(self, QVisionWidget):
        _translate = QtCore.QCoreApplication.translate
        QVisionWidget.setWindowTitle(_translate("QVisionWidget", "Form"))
        self.groupProcess.setTitle(_translate("QVisionWidget", "Processing options"))
        self.breal.setText(_translate("QVisionWidget", "Real-time"))
        self.bpost.setText(_translate("QVisionWidget", "Post-process"))
        self.skipBox.setSuffix(_translate("QVisionWidget", " frames"))
        self.skipBox.setPrefix(_translate("QVisionWidget", "Skip "))
        self.groupPipeline.setTitle(_translate("QVisionWidget", "Vision pipeline"))
        self.bDetect.setText(_translate("QVisionWidget", "Detect"))
        self.bEstimate.setText(_translate("QVisionWidget", "Estimate"))
        self.bRefine.setText(_translate("QVisionWidget", "Refine"))
        self.groupExport.setTitle(_translate("QVisionWidget", "Export options"))
        self.checkFrames.setText(_translate("QVisionWidget", "Save frames"))
        self.checkTrajectories.setText(_translate("QVisionWidget", "Save trajectories"))

from pyqtgraph import PlotWidget

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QVisionWidget = QtWidgets.QWidget()
    ui = Ui_QVisionWidget()
    ui.setupUi(QVisionWidget)
    QVisionWidget.show()
    sys.exit(app.exec_())

