# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'FabWidget.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PyFab(object):
    def setupUi(self, PyFab):
        PyFab.setObjectName("PyFab")
        PyFab.resize(1072, 732)
        self.centralwidget = QtWidgets.QWidget(PyFab)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setContentsMargins(2, 2, 2, 2)
        self.gridLayout.setSpacing(2)
        self.gridLayout.setObjectName("gridLayout")
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName("splitter")
        self.screen = QJansenScreen(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.screen.sizePolicy().hasHeightForWidth())
        self.screen.setSizePolicy(sizePolicy)
        self.screen.setMinimumSize(QtCore.QSize(640, 480))
        self.screen.setObjectName("screen")
        self.tabWidget = QtWidgets.QTabWidget(self.splitter)
        self.tabWidget.setMinimumSize(QtCore.QSize(400, 0))
        self.tabWidget.setAccessibleName("")
        self.tabWidget.setObjectName("tabWidget")
        self.tabVideo = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabVideo.sizePolicy().hasHeightForWidth())
        self.tabVideo.setSizePolicy(sizePolicy)
        self.tabVideo.setMinimumSize(QtCore.QSize(0, 0))
        self.tabVideo.setAccessibleName("")
        self.tabVideo.setAccessibleDescription("")
        self.tabVideo.setObjectName("tabVideo")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tabVideo)
        self.verticalLayout.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout.setSpacing(2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupSource = QtWidgets.QGroupBox(self.tabVideo)
        self.groupSource.setObjectName("groupSource")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.groupSource)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.bcamera = QtWidgets.QRadioButton(self.groupSource)
        self.bcamera.setChecked(True)
        self.bcamera.setObjectName("bcamera")
        self.horizontalLayout.addWidget(self.bcamera)
        self.bfilters = QtWidgets.QRadioButton(self.groupSource)
        self.bfilters.setObjectName("bfilters")
        self.horizontalLayout.addWidget(self.bfilters)
        spacerItem = QtWidgets.QSpacerItem(283, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.label = QtWidgets.QLabel(self.groupSource)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.fps = QtWidgets.QDoubleSpinBox(self.groupSource)
        self.fps.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.fps.setDecimals(1)
        self.fps.setMaximum(1000.0)
        self.fps.setProperty("value", 10.0)
        self.fps.setObjectName("fps")
        self.horizontalLayout.addWidget(self.fps)
        self.verticalLayout.addWidget(self.groupSource)
        self.groupDvr = QtWidgets.QGroupBox(self.tabVideo)
        self.groupDvr.setMinimumSize(QtCore.QSize(0, 0))
        self.groupDvr.setObjectName("groupDvr")
        self.dvrLayout = QtWidgets.QVBoxLayout(self.groupDvr)
        self.dvrLayout.setContentsMargins(2, 2, 2, 2)
        self.dvrLayout.setSpacing(1)
        self.dvrLayout.setObjectName("dvrLayout")
        self.dvr = QDVR(self.groupDvr)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.dvr.sizePolicy().hasHeightForWidth())
        self.dvr.setSizePolicy(sizePolicy)
        self.dvr.setMinimumSize(QtCore.QSize(0, 0))
        self.dvr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.dvr.setFrameShadow(QtWidgets.QFrame.Raised)
        self.dvr.setObjectName("dvr")
        self.dvrLayout.addWidget(self.dvr)
        self.verticalLayout.addWidget(self.groupDvr)
        self.groupCamera = QtWidgets.QGroupBox(self.tabVideo)
        self.groupCamera.setMinimumSize(QtCore.QSize(0, 0))
        self.groupCamera.setObjectName("groupCamera")
        self.cameraLayout = QtWidgets.QVBoxLayout(self.groupCamera)
        self.cameraLayout.setContentsMargins(2, 2, 2, 2)
        self.cameraLayout.setSpacing(1)
        self.cameraLayout.setObjectName("cameraLayout")
        self.camera = QtWidgets.QFrame(self.groupCamera)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.camera.sizePolicy().hasHeightForWidth())
        self.camera.setSizePolicy(sizePolicy)
        self.camera.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.camera.setFrameShadow(QtWidgets.QFrame.Raised)
        self.camera.setObjectName("camera")
        self.cameraLayout.addWidget(self.camera)
        self.verticalLayout.addWidget(self.groupCamera)
        self.groupFilters = QtWidgets.QGroupBox(self.tabVideo)
        self.groupFilters.setMinimumSize(QtCore.QSize(0, 0))
        self.groupFilters.setObjectName("groupFilters")
        self.filterLayout = QtWidgets.QVBoxLayout(self.groupFilters)
        self.filterLayout.setContentsMargins(2, 2, 2, 2)
        self.filterLayout.setSpacing(1)
        self.filterLayout.setObjectName("filterLayout")
        self.filters = QVideoFilter(self.groupFilters)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.filters.sizePolicy().hasHeightForWidth())
        self.filters.setSizePolicy(sizePolicy)
        self.filters.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.filters.setFrameShadow(QtWidgets.QFrame.Raised)
        self.filters.setObjectName("filters")
        self.filterLayout.addWidget(self.filters)
        self.verticalLayout.addWidget(self.groupFilters)
        spacerItem1 = QtWidgets.QSpacerItem(20, 338, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.tabWidget.addTab(self.tabVideo, "")
        self.tabHistogram = QtWidgets.QWidget()
        self.tabHistogram.setObjectName("tabHistogram")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tabHistogram)
        self.verticalLayout_3.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_3.setSpacing(2)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.histogram = QHistogram(self.tabHistogram)
        self.histogram.setObjectName("histogram")
        self.verticalLayout_3.addWidget(self.histogram)
        self.tabWidget.addTab(self.tabHistogram, "")
        self.tabHardware = QtWidgets.QWidget()
        self.tabHardware.setObjectName("tabHardware")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tabHardware)
        self.verticalLayout_4.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_4.setSpacing(2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.hardware = QHardwareTab(self.tabHardware)
        self.hardware.setObjectName("hardware")
        self.verticalLayout_4.addWidget(self.hardware)
        self.tabWidget.addTab(self.tabHardware, "")
        self.tabTraps = QtWidgets.QWidget()
        self.tabTraps.setObjectName("tabTraps")
        self.verticalLayoutTraps = QtWidgets.QVBoxLayout(self.tabTraps)
        self.verticalLayoutTraps.setContentsMargins(2, 2, 2, 2)
        self.verticalLayoutTraps.setSpacing(2)
        self.verticalLayoutTraps.setObjectName("verticalLayoutTraps")
        self.traps = QTrapWidget(self.tabTraps)
        self.traps.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.traps.setFrameShadow(QtWidgets.QFrame.Raised)
        self.traps.setObjectName("traps")
        self.verticalLayoutTraps.addWidget(self.traps)
        self.tabWidget.addTab(self.tabTraps, "")
        self.tabCGH = QtWidgets.QWidget()
        self.tabCGH.setObjectName("tabCGH")
        self.verticalLayoutCGH = QtWidgets.QVBoxLayout(self.tabCGH)
        self.verticalLayoutCGH.setContentsMargins(2, 2, 2, 2)
        self.verticalLayoutCGH.setSpacing(2)
        self.verticalLayoutCGH.setObjectName("verticalLayoutCGH")
        self.cgh = QCGH(self.tabCGH)
        self.cgh.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cgh.setFrameShadow(QtWidgets.QFrame.Raised)
        self.cgh.setObjectName("cgh")
        self.verticalLayoutCGH.addWidget(self.cgh)
        spacerItem2 = QtWidgets.QSpacerItem(20, 511, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayoutCGH.addItem(spacerItem2)
        self.tabWidget.addTab(self.tabCGH, "")
        self.tabSLM = QtWidgets.QWidget()
        self.tabSLM.setObjectName("tabSLM")
        self.verticalLayoutSLM = QtWidgets.QVBoxLayout(self.tabSLM)
        self.verticalLayoutSLM.setContentsMargins(2, 2, 2, 2)
        self.verticalLayoutSLM.setSpacing(2)
        self.verticalLayoutSLM.setObjectName("verticalLayoutSLM")
        self.slmView = QSLMWidget(self.tabSLM)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slmView.sizePolicy().hasHeightForWidth())
        self.slmView.setSizePolicy(sizePolicy)
        self.slmView.setObjectName("slmView")
        self.verticalLayoutSLM.addWidget(self.slmView)
        self.tabWidget.addTab(self.tabSLM, "")
        self.tabHelp = QtWidgets.QWidget()
        self.tabHelp.setObjectName("tabHelp")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tabHelp)
        self.verticalLayout_2.setContentsMargins(2, 2, 2, 2)
        self.verticalLayout_2.setSpacing(4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.bback = QtWidgets.QPushButton(self.tabHelp)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bback.sizePolicy().hasHeightForWidth())
        self.bback.setSizePolicy(sizePolicy)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/icons/icons/go-previous.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.bback.setIcon(icon)
        self.bback.setObjectName("bback")
        self.verticalLayout_2.addWidget(self.bback)
        self.browser = QtWebKitWidgets.QWebView(self.tabHelp)
        self.browser.setUrl(QtCore.QUrl("qrc:/help/help/pyfab.html"))
        self.browser.setZoomFactor(1.25)
        self.browser.setObjectName("browser")
        self.verticalLayout_2.addWidget(self.browser)
        self.tabWidget.addTab(self.tabHelp, "")
        self.gridLayout.addWidget(self.splitter, 0, 0, 1, 1)
        PyFab.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(PyFab)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1072, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTasks = QtWidgets.QMenu(self.menubar)
        self.menuTasks.setObjectName("menuTasks")
        PyFab.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(PyFab)
        self.statusbar.setObjectName("statusbar")
        PyFab.setStatusBar(self.statusbar)
        self.actionSavePhoto = QtWidgets.QAction(PyFab)
        self.actionSavePhoto.setObjectName("actionSavePhoto")
        self.actionSavePhotoAs = QtWidgets.QAction(PyFab)
        self.actionSavePhotoAs.setObjectName("actionSavePhotoAs")
        self.actionQuit = QtWidgets.QAction(PyFab)
        self.actionQuit.setObjectName("actionQuit")
        self.actionSaveHologram = QtWidgets.QAction(PyFab)
        self.actionSaveHologram.setObjectName("actionSaveHologram")
        self.actionSaveSettings = QtWidgets.QAction(PyFab)
        self.actionSaveSettings.setObjectName("actionSaveSettings")
        self.actionPauseTasks = QtWidgets.QAction(PyFab)
        self.actionPauseTasks.setObjectName("actionPauseTasks")
        self.actionStopTasks = QtWidgets.QAction(PyFab)
        self.actionStopTasks.setObjectName("actionStopTasks")
        self.actionSaveHologramAs = QtWidgets.QAction(PyFab)
        self.actionSaveHologramAs.setObjectName("actionSaveHologramAs")
        self.menuFile.addAction(self.actionSavePhoto)
        self.menuFile.addAction(self.actionSavePhotoAs)
        self.menuFile.addAction(self.actionSaveHologram)
        self.menuFile.addAction(self.actionSaveHologramAs)
        self.menuFile.addAction(self.actionSaveSettings)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuTasks.addAction(self.actionPauseTasks)
        self.menuTasks.addAction(self.actionStopTasks)
        self.menuTasks.addSeparator()
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTasks.menuAction())

        self.retranslateUi(PyFab)
        self.tabWidget.setCurrentIndex(6)
        self.actionQuit.triggered.connect(PyFab.close)
        self.bback.clicked.connect(self.browser.back)
        self.dvr.recording['bool'].connect(self.camera.setDisabled)
        self.dvr.recording['bool'].connect(self.filters.setDisabled)
        self.actionPauseTasks.triggered.connect(PyFab.pauseTasks)
        self.actionStopTasks.triggered.connect(PyFab.stopTasks)
        self.actionSaveSettings.triggered.connect(PyFab.saveSettings)
        self.actionSavePhoto.triggered.connect(PyFab.savePhoto)
        self.actionSavePhotoAs.triggered.connect(PyFab.savePhotoAs)
        self.actionSaveHologram.triggered.connect(PyFab.saveHologram)
        self.actionSaveHologramAs.triggered.connect(PyFab.saveHologramAs)
        self.tabWidget.currentChanged['int'].connect(self.hardware.expose)
        self.screen.sigFPS['double'].connect(self.fps.setValue)
        QtCore.QMetaObject.connectSlotsByName(PyFab)

    def retranslateUi(self, PyFab):
        _translate = QtCore.QCoreApplication.translate
        PyFab.setWindowTitle(_translate("PyFab", "pyFab"))
        self.tabWidget.setStatusTip(_translate("PyFab", "Record from video camera directly"))
        self.groupSource.setTitle(_translate("PyFab", "Recording Source"))
        self.bcamera.setStatusTip(_translate("PyFab", "Record raw video from camera"))
        self.bcamera.setText(_translate("PyFab", "Camera"))
        self.bfilters.setStatusTip(_translate("PyFab", "Record filtered video stream"))
        self.bfilters.setText(_translate("PyFab", "Filters"))
        self.label.setText(_translate("PyFab", "Rate"))
        self.fps.setSuffix(_translate("PyFab", " fps"))
        self.groupDvr.setTitle(_translate("PyFab", "Video Recorder"))
        self.groupCamera.setTitle(_translate("PyFab", "Video Camera"))
        self.groupFilters.setTitle(_translate("PyFab", "Video Filters"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabVideo), _translate("PyFab", "Video"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabHistogram), _translate("PyFab", "Histogram"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabHardware), _translate("PyFab", "Hardware"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabTraps), _translate("PyFab", "Traps"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabCGH), _translate("PyFab", "CGH"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabSLM), _translate("PyFab", "SLM"))
        self.bback.setText(_translate("PyFab", "Back"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabHelp), _translate("PyFab", "Help"))
        self.menuFile.setTitle(_translate("PyFab", "&File"))
        self.menuTasks.setTitle(_translate("PyFab", "&Tasks"))
        self.actionSavePhoto.setText(_translate("PyFab", "&Save Photo"))
        self.actionSavePhoto.setShortcut(_translate("PyFab", "Ctrl+S"))
        self.actionSavePhotoAs.setText(_translate("PyFab", "Save Photo &As ..."))
        self.actionSavePhotoAs.setShortcut(_translate("PyFab", "Ctrl+A"))
        self.actionQuit.setText(_translate("PyFab", "&Quit"))
        self.actionQuit.setShortcut(_translate("PyFab", "Ctrl+Q"))
        self.actionSaveHologram.setText(_translate("PyFab", "Save &Hologram"))
        self.actionSaveHologram.setShortcut(_translate("PyFab", "Ctrl+H"))
        self.actionSaveSettings.setText(_translate("PyFab", "Save Se&ttings"))
        self.actionSaveSettings.setShortcut(_translate("PyFab", "Ctrl+T"))
        self.actionPauseTasks.setText(_translate("PyFab", "&Pause Tasks"))
        self.actionPauseTasks.setStatusTip(_translate("PyFab", "Pause/resume tasks"))
        self.actionPauseTasks.setShortcut(_translate("PyFab", "Ctrl+P"))
        self.actionStopTasks.setText(_translate("PyFab", "Stop T&asks"))
        self.actionStopTasks.setStatusTip(_translate("PyFab", "Empty task queue"))
        self.actionStopTasks.setShortcut(_translate("PyFab", "Ctrl+A"))
        self.actionSaveHologramAs.setText(_translate("PyFab", "Sa&ve Hologram As ..."))

from PyQt5 import QtWebKitWidgets
from jansenlib.QDVR.QDVR import QDVR
from jansenlib.QHistogram.QHistogram import QHistogram
from jansenlib.QJansenScreen import QJansenScreen
from jansenlib.video.QVideoFilter.QVideoFilter import QVideoFilter
from pyfablib.QCGH.QCGH import QCGH
from pyfablib.QHardwareTab import QHardwareTab
from pyfablib.QSLMWidget import QSLMWidget
from pyfablib.traps.QTrapWidget import QTrapWidget
import help_rc

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    PyFab = QtWidgets.QMainWindow()
    ui = Ui_PyFab()
    ui.setupUi(PyFab)
    PyFab.show()
    sys.exit(app.exec_())

