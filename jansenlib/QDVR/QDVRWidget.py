# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QDVRWidget.ui'
#
# Created by: PyQt5 UI code generator 5.11.3
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QDVRWidget(object):
    def setupUi(self, QDVRWidget):
        QDVRWidget.setObjectName("QDVRWidget")
        QDVRWidget.resize(215, 120)
        QDVRWidget.setMinimumSize(QtCore.QSize(215, 120))
        font = QtGui.QFont()
        font.setFamily("Arial")
        QDVRWidget.setFont(font)
        QDVRWidget.setFrameShape(QtWidgets.QFrame.Box)
        self.verticalLayout = QtWidgets.QVBoxLayout(QDVRWidget)
        self.verticalLayout.setContentsMargins(2, 2, 2, 4)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.widgetRecord = QtWidgets.QWidget(QDVRWidget)
        self.widgetRecord.setObjectName("widgetRecord")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.widgetRecord)
        self.horizontalLayout_2.setContentsMargins(0, 0, 6, 0)
        self.horizontalLayout_2.setSpacing(2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.recordButton = QtWidgets.QPushButton(self.widgetRecord)
        self.recordButton.setObjectName("recordButton")
        self.horizontalLayout_2.addWidget(self.recordButton)
        self.stopButton = QtWidgets.QPushButton(self.widgetRecord)
        self.stopButton.setObjectName("stopButton")
        self.horizontalLayout_2.addWidget(self.stopButton)
        self.frameNumber = QtWidgets.QLCDNumber(self.widgetRecord)
        self.frameNumber.setAutoFillBackground(False)
        self.frameNumber.setStyleSheet("QLCDNumber{\n"
"    color: rgb(0, 0, 0);    \n"
"    background-color: rgb(255, 255, 255);\n"
"}")
        self.frameNumber.setSegmentStyle(QtWidgets.QLCDNumber.Flat)
        self.frameNumber.setObjectName("frameNumber")
        self.horizontalLayout_2.addWidget(self.frameNumber)
        self.verticalLayout.addWidget(self.widgetRecord)
        self.widgetSaveFile = QtWidgets.QWidget(QDVRWidget)
        self.widgetSaveFile.setObjectName("widgetSaveFile")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widgetSaveFile)
        self.horizontalLayout_3.setContentsMargins(6, 0, 6, 0)
        self.horizontalLayout_3.setSpacing(6)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.saveLabel = QtWidgets.QLabel(self.widgetSaveFile)
        self.saveLabel.setObjectName("saveLabel")
        self.horizontalLayout_3.addWidget(self.saveLabel)
        self.saveEdit = QtWidgets.QLineEdit(self.widgetSaveFile)
        self.saveEdit.setReadOnly(True)
        self.saveEdit.setObjectName("saveEdit")
        self.horizontalLayout_3.addWidget(self.saveEdit)
        self.verticalLayout.addWidget(self.widgetSaveFile)
        self.widgetPlay = QtWidgets.QWidget(QDVRWidget)
        self.widgetPlay.setObjectName("widgetPlay")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widgetPlay)
        self.horizontalLayout_4.setContentsMargins(0, 1, 0, 1)
        self.horizontalLayout_4.setSpacing(2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.rewindButton = QtWidgets.QPushButton(self.widgetPlay)
        self.rewindButton.setObjectName("rewindButton")
        self.horizontalLayout_4.addWidget(self.rewindButton)
        self.pauseButton = QtWidgets.QPushButton(self.widgetPlay)
        self.pauseButton.setObjectName("pauseButton")
        self.horizontalLayout_4.addWidget(self.pauseButton)
        self.playButton = QtWidgets.QPushButton(self.widgetPlay)
        self.playButton.setObjectName("playButton")
        self.horizontalLayout_4.addWidget(self.playButton)
        self.verticalLayout.addWidget(self.widgetPlay)
        self.widgetPlayFile = QtWidgets.QWidget(QDVRWidget)
        self.widgetPlayFile.setObjectName("widgetPlayFile")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widgetPlayFile)
        self.horizontalLayout_5.setContentsMargins(6, 0, 6, 0)
        self.horizontalLayout_5.setSpacing(6)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.labelPlayFile = QtWidgets.QLabel(self.widgetPlayFile)
        self.labelPlayFile.setObjectName("labelPlayFile")
        self.horizontalLayout_5.addWidget(self.labelPlayFile)
        self.playEdit = QtWidgets.QLineEdit(self.widgetPlayFile)
        self.playEdit.setReadOnly(True)
        self.playEdit.setObjectName("playEdit")
        self.horizontalLayout_5.addWidget(self.playEdit)
        self.verticalLayout.addWidget(self.widgetPlayFile)
        self.saveLabel.setBuddy(self.saveEdit)
        self.labelPlayFile.setBuddy(self.playEdit)

        self.retranslateUi(QDVRWidget)
        QtCore.QMetaObject.connectSlotsByName(QDVRWidget)

    def retranslateUi(self, QDVRWidget):
        _translate = QtCore.QCoreApplication.translate
        QDVRWidget.setWindowTitle(_translate("QDVRWidget", "QDVRWidget"))
        QDVRWidget.setStatusTip(_translate("QDVRWidget", "Video Recorder"))
        self.recordButton.setStatusTip(_translate("QDVRWidget", "Record video"))
        self.recordButton.setText(_translate("QDVRWidget", "Record"))
        self.stopButton.setStatusTip(_translate("QDVRWidget", "Stop recording"))
        self.stopButton.setText(_translate("QDVRWidget", "Stop"))
        self.saveLabel.setText(_translate("QDVRWidget", "Save As"))
        self.saveEdit.setStatusTip(_translate("QDVRWidget", "Video file name"))
        self.rewindButton.setStatusTip(_translate("QDVRWidget", "Rewind video file"))
        self.rewindButton.setText(_translate("QDVRWidget", "Rewind"))
        self.pauseButton.setStatusTip(_translate("QDVRWidget", "Pause video playback"))
        self.pauseButton.setText(_translate("QDVRWidget", "Pause"))
        self.playButton.setStatusTip(_translate("QDVRWidget", "Play video file"))
        self.playButton.setText(_translate("QDVRWidget", "Play"))
        self.labelPlayFile.setText(_translate("QDVRWidget", "Play"))
        self.playEdit.setStatusTip(_translate("QDVRWidget", "Video file"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QDVRWidget = QtWidgets.QFrame()
    ui = Ui_QDVRWidget()
    ui.setupUi(QDVRWidget)
    QDVRWidget.show()
    sys.exit(app.exec_())

