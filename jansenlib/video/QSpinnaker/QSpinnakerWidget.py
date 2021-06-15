# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QSpinnakerWidget.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_QSpinnakerWidget(object):
    def setupUi(self, QSpinnakerWidget):
        QSpinnakerWidget.setObjectName("QSpinnakerWidget")
        QSpinnakerWidget.resize(505, 244)
        QSpinnakerWidget.setMinimumSize(QtCore.QSize(248, 244))
        self.verticalLayout = QtWidgets.QVBoxLayout(QSpinnakerWidget)
        self.verticalLayout.setContentsMargins(2, 1, 2, 1)
        self.verticalLayout.setSpacing(1)
        self.verticalLayout.setObjectName("verticalLayout")
        self.cameraname = QtWidgets.QLabel(QSpinnakerWidget)
        self.cameraname.setObjectName("cameraname")
        self.verticalLayout.addWidget(self.cameraname)
        self.frameCheckBoxes = QtWidgets.QFrame(QSpinnakerWidget)
        self.frameCheckBoxes.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameCheckBoxes.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameCheckBoxes.setObjectName("frameCheckBoxes")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frameCheckBoxes)
        self.horizontalLayout.setContentsMargins(2, 1, 2, 1)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.gray = QtWidgets.QCheckBox(self.frameCheckBoxes)
        self.gray.setObjectName("gray")
        self.horizontalLayout.addWidget(self.gray)
        self.mirrored = QtWidgets.QCheckBox(self.frameCheckBoxes)
        self.mirrored.setObjectName("mirrored")
        self.horizontalLayout.addWidget(self.mirrored)
        self.flipped = QtWidgets.QCheckBox(self.frameCheckBoxes)
        self.flipped.setObjectName("flipped")
        self.horizontalLayout.addWidget(self.flipped)
        self.verticalLayout.addWidget(self.frameCheckBoxes)
        self.frameExposure = QtWidgets.QFrame(QSpinnakerWidget)
        self.frameExposure.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameExposure.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameExposure.setObjectName("frameExposure")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frameExposure)
        self.gridLayout_2.setContentsMargins(2, 1, 2, 1)
        self.gridLayout_2.setHorizontalSpacing(2)
        self.gridLayout_2.setVerticalSpacing(1)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.labelgamma = QtWidgets.QLabel(self.frameExposure)
        self.labelgamma.setObjectName("labelgamma")
        self.gridLayout_2.addWidget(self.labelgamma, 5, 0, 1, 1)
        self.exposureLabel = QtWidgets.QLabel(self.frameExposure)
        self.exposureLabel.setObjectName("exposureLabel")
        self.gridLayout_2.addWidget(self.exposureLabel, 1, 0, 1, 1)
        self.autoexposure = QtWidgets.QPushButton(self.frameExposure)
        self.autoexposure.setObjectName("autoexposure")
        self.gridLayout_2.addWidget(self.autoexposure, 1, 2, 1, 1)
        self.labelblacklevel = QtWidgets.QLabel(self.frameExposure)
        self.labelblacklevel.setObjectName("labelblacklevel")
        self.gridLayout_2.addWidget(self.labelblacklevel, 4, 0, 1, 1)
        self.framerate = QtWidgets.QDoubleSpinBox(self.frameExposure)
        self.framerate.setDecimals(1)
        self.framerate.setMinimum(1.0)
        self.framerate.setMaximum(40.0)
        self.framerate.setProperty("value", 40.0)
        self.framerate.setObjectName("framerate")
        self.gridLayout_2.addWidget(self.framerate, 0, 1, 1, 1)
        self.gamma = QtWidgets.QDoubleSpinBox(self.frameExposure)
        self.gamma.setMinimum(0.5)
        self.gamma.setMaximum(4.0)
        self.gamma.setSingleStep(0.1)
        self.gamma.setProperty("value", 1.0)
        self.gamma.setObjectName("gamma")
        self.gridLayout_2.addWidget(self.gamma, 5, 1, 1, 1)
        self.labelFrameRate = QtWidgets.QLabel(self.frameExposure)
        self.labelFrameRate.setObjectName("labelFrameRate")
        self.gridLayout_2.addWidget(self.labelFrameRate, 0, 0, 1, 1)
        self.exposuretime = QtWidgets.QDoubleSpinBox(self.frameExposure)
        self.exposuretime.setDecimals(0)
        self.exposuretime.setMinimum(10.0)
        self.exposuretime.setMaximum(50000.0)
        self.exposuretime.setSingleStep(10.0)
        self.exposuretime.setObjectName("exposuretime")
        self.gridLayout_2.addWidget(self.exposuretime, 1, 1, 1, 1)
        self.autogain = QtWidgets.QPushButton(self.frameExposure)
        self.autogain.setObjectName("autogain")
        self.gridLayout_2.addWidget(self.autogain, 2, 2, 1, 1)
        self.gain = QtWidgets.QDoubleSpinBox(self.frameExposure)
        self.gain.setDecimals(1)
        self.gain.setMaximum(24.0)
        self.gain.setSingleStep(0.1)
        self.gain.setObjectName("gain")
        self.gridLayout_2.addWidget(self.gain, 2, 1, 1, 1)
        self.gainLabel = QtWidgets.QLabel(self.frameExposure)
        self.gainLabel.setObjectName("gainLabel")
        self.gridLayout_2.addWidget(self.gainLabel, 2, 0, 1, 1)
        self.blacklevel = QtWidgets.QSpinBox(self.frameExposure)
        self.blacklevel.setObjectName("blacklevel")
        self.gridLayout_2.addWidget(self.blacklevel, 4, 1, 1, 1)
        self.verticalLayout.addWidget(self.frameExposure)
        self.frameGeometry = QtWidgets.QFrame(QSpinnakerWidget)
        self.frameGeometry.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frameGeometry.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frameGeometry.setObjectName("frameGeometry")
        self.gridLayout = QtWidgets.QGridLayout(self.frameGeometry)
        self.gridLayout.setContentsMargins(2, 1, 2, 1)
        self.gridLayout.setHorizontalSpacing(2)
        self.gridLayout.setVerticalSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        self.labely0 = QtWidgets.QLabel(self.frameGeometry)
        self.labely0.setObjectName("labely0")
        self.gridLayout.addWidget(self.labely0, 1, 1, 1, 1)
        self.x0 = QtWidgets.QSpinBox(self.frameGeometry)
        self.x0.setObjectName("x0")
        self.gridLayout.addWidget(self.x0, 3, 0, 1, 1)
        self.height = QtWidgets.QSpinBox(self.frameGeometry)
        self.height.setMinimum(16)
        self.height.setMaximum(1080)
        self.height.setSingleStep(16)
        self.height.setProperty("value", 1080)
        self.height.setObjectName("height")
        self.gridLayout.addWidget(self.height, 3, 4, 1, 1)
        self.labelheight = QtWidgets.QLabel(self.frameGeometry)
        self.labelheight.setObjectName("labelheight")
        self.gridLayout.addWidget(self.labelheight, 1, 4, 1, 1)
        self.y0 = QtWidgets.QSpinBox(self.frameGeometry)
        self.y0.setObjectName("y0")
        self.gridLayout.addWidget(self.y0, 3, 1, 1, 1)
        self.width = QtWidgets.QSpinBox(self.frameGeometry)
        self.width.setMinimum(16)
        self.width.setMaximum(1280)
        self.width.setSingleStep(16)
        self.width.setProperty("value", 1280)
        self.width.setObjectName("width")
        self.gridLayout.addWidget(self.width, 3, 3, 1, 1)
        self.labelx0 = QtWidgets.QLabel(self.frameGeometry)
        self.labelx0.setObjectName("labelx0")
        self.gridLayout.addWidget(self.labelx0, 1, 0, 1, 1)
        self.labelwidth = QtWidgets.QLabel(self.frameGeometry)
        self.labelwidth.setObjectName("labelwidth")
        self.gridLayout.addWidget(self.labelwidth, 1, 3, 1, 1)
        self.verticalLayout.addWidget(self.frameGeometry)
        self.labelgamma.setBuddy(self.gamma)
        self.exposureLabel.setBuddy(self.exposuretime)
        self.labelblacklevel.setBuddy(self.blacklevel)
        self.labelFrameRate.setBuddy(self.framerate)
        self.gainLabel.setBuddy(self.gain)
        self.labely0.setBuddy(self.y0)
        self.labelheight.setBuddy(self.height)
        self.labelx0.setBuddy(self.x0)
        self.labelwidth.setBuddy(self.width)

        self.retranslateUi(QSpinnakerWidget)
        QtCore.QMetaObject.connectSlotsByName(QSpinnakerWidget)
        QSpinnakerWidget.setTabOrder(self.gray, self.mirrored)
        QSpinnakerWidget.setTabOrder(self.mirrored, self.flipped)
        QSpinnakerWidget.setTabOrder(self.flipped, self.framerate)
        QSpinnakerWidget.setTabOrder(self.framerate, self.exposuretime)
        QSpinnakerWidget.setTabOrder(self.exposuretime, self.autoexposure)
        QSpinnakerWidget.setTabOrder(self.autoexposure, self.gain)
        QSpinnakerWidget.setTabOrder(self.gain, self.autogain)
        QSpinnakerWidget.setTabOrder(self.autogain, self.blacklevel)
        QSpinnakerWidget.setTabOrder(self.blacklevel, self.gamma)
        QSpinnakerWidget.setTabOrder(self.gamma, self.x0)
        QSpinnakerWidget.setTabOrder(self.x0, self.y0)
        QSpinnakerWidget.setTabOrder(self.y0, self.width)
        QSpinnakerWidget.setTabOrder(self.width, self.height)

    def retranslateUi(self, QSpinnakerWidget):
        _translate = QtCore.QCoreApplication.translate
        QSpinnakerWidget.setWindowTitle(_translate("QSpinnakerWidget", "QSpinnakerWidget"))
        QSpinnakerWidget.setStatusTip(_translate("QSpinnakerWidget", "Control Spinnaker camera"))
        self.cameraname.setText(_translate("QSpinnakerWidget", "Camera Name"))
        self.gray.setText(_translate("QSpinnakerWidget", "Gray"))
        self.mirrored.setStatusTip(_translate("QSpinnakerWidget", "Camera: Flip image around vertical axis"))
        self.mirrored.setText(_translate("QSpinnakerWidget", "&Mirrored"))
        self.flipped.setStatusTip(_translate("QSpinnakerWidget", "Camera: Flip image about horizontal axis"))
        self.flipped.setText(_translate("QSpinnakerWidget", "&Flipped"))
        self.labelgamma.setText(_translate("QSpinnakerWidget", "Gamma"))
        self.exposureLabel.setText(_translate("QSpinnakerWidget", "&Exposure Time"))
        self.autoexposure.setStatusTip(_translate("QSpinnakerWidget", "Camera: Optimize exposure time"))
        self.autoexposure.setText(_translate("QSpinnakerWidget", "Auto"))
        self.labelblacklevel.setText(_translate("QSpinnakerWidget", "Black Level"))
        self.framerate.setStatusTip(_translate("QSpinnakerWidget", "Camera frame rate"))
        self.framerate.setSuffix(_translate("QSpinnakerWidget", " Hz"))
        self.gamma.setStatusTip(_translate("QSpinnakerWidget", "Camera gamma"))
        self.labelFrameRate.setText(_translate("QSpinnakerWidget", "Frame &Rate"))
        self.exposuretime.setStatusTip(_translate("QSpinnakerWidget", "Camera exposure time "))
        self.exposuretime.setSuffix(_translate("QSpinnakerWidget", " μs"))
        self.autogain.setStatusTip(_translate("QSpinnakerWidget", "Camera: Optimize gain"))
        self.autogain.setText(_translate("QSpinnakerWidget", "Auto"))
        self.gain.setStatusTip(_translate("QSpinnakerWidget", "Camera gain"))
        self.gain.setSuffix(_translate("QSpinnakerWidget", " dB"))
        self.gainLabel.setText(_translate("QSpinnakerWidget", "&Gain"))
        self.blacklevel.setStatusTip(_translate("QSpinnakerWidget", "Camera black level"))
        self.labely0.setText(_translate("QSpinnakerWidget", "&y0"))
        self.x0.setStatusTip(_translate("QSpinnakerWidget", "Camera ROI: bottom left corner"))
        self.height.setStatusTip(_translate("QSpinnakerWidget", "Camera ROI: height"))
        self.labelheight.setText(_translate("QSpinnakerWidget", "&Height"))
        self.y0.setStatusTip(_translate("QSpinnakerWidget", "Camera ROI: bottom left corner"))
        self.width.setStatusTip(_translate("QSpinnakerWidget", "Camera ROI: width"))
        self.labelx0.setText(_translate("QSpinnakerWidget", "&x0"))
        self.labelwidth.setText(_translate("QSpinnakerWidget", "&Width"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QSpinnakerWidget = QtWidgets.QFrame()
    ui = Ui_QSpinnakerWidget()
    ui.setupUi(QSpinnakerWidget)
    QSpinnakerWidget.show()
    sys.exit(app.exec_())
