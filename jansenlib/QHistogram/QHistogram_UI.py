# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QHistogram_UI.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_QHistogramWidget(object):
    def setupUi(self, QHistogramWidget):
        QHistogramWidget.setObjectName("QHistogramWidget")
        QHistogramWidget.resize(550, 742)
        self.verticalLayoutHistogram = QtWidgets.QVBoxLayout(QHistogramWidget)
        self.verticalLayoutHistogram.setContentsMargins(2, 2, 2, 2)
        self.verticalLayoutHistogram.setSpacing(2)
        self.verticalLayoutHistogram.setObjectName("verticalLayoutHistogram")
        self.groupHistogram = QtWidgets.QGroupBox(QHistogramWidget)
        self.groupHistogram.setObjectName("groupHistogram")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupHistogram)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.histo = PlotWidget(self.groupHistogram)
        self.histo.setObjectName("histo")
        self.verticalLayout.addWidget(self.histo)
        self.verticalLayoutHistogram.addWidget(self.groupHistogram)
        self.groupHorizontal = QtWidgets.QGroupBox(QHistogramWidget)
        self.groupHorizontal.setObjectName("groupHorizontal")
        self.verticalLayoutHProfile = QtWidgets.QVBoxLayout(self.groupHorizontal)
        self.verticalLayoutHProfile.setContentsMargins(2, 2, 2, 2)
        self.verticalLayoutHProfile.setSpacing(2)
        self.verticalLayoutHProfile.setObjectName("verticalLayoutHProfile")
        self.xmean = PlotWidget(self.groupHorizontal)
        self.xmean.setObjectName("xmean")
        self.verticalLayoutHProfile.addWidget(self.xmean)
        self.verticalLayoutHistogram.addWidget(self.groupHorizontal)
        self.groupVertical = QtWidgets.QGroupBox(QHistogramWidget)
        self.groupVertical.setObjectName("groupVertical")
        self.verticalLayoutVProfile = QtWidgets.QVBoxLayout(self.groupVertical)
        self.verticalLayoutVProfile.setContentsMargins(2, 2, 2, 2)
        self.verticalLayoutVProfile.setSpacing(2)
        self.verticalLayoutVProfile.setObjectName("verticalLayoutVProfile")
        self.ymean = PlotWidget(self.groupVertical)
        self.ymean.setObjectName("ymean")
        self.verticalLayoutVProfile.addWidget(self.ymean)
        self.verticalLayoutHistogram.addWidget(self.groupVertical)

        self.retranslateUi(QHistogramWidget)
        QtCore.QMetaObject.connectSlotsByName(QHistogramWidget)

    def retranslateUi(self, QHistogramWidget):
        _translate = QtCore.QCoreApplication.translate
        QHistogramWidget.setWindowTitle(_translate("QHistogramWidget", "Histogram"))
        self.groupHistogram.setTitle(_translate("QHistogramWidget", "Histogram"))
        self.groupHorizontal.setTitle(_translate("QHistogramWidget", "Horizontal Profile"))
        self.groupVertical.setTitle(_translate("QHistogramWidget", "Vertical Profile"))

from pyqtgraph import PlotWidget

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    QHistogramWidget = QtWidgets.QWidget()
    ui = Ui_QHistogramWidget()
    ui.setupUi(QHistogramWidget)
    QHistogramWidget.show()
    sys.exit(app.exec_())

