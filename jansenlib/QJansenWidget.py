# -*- coding: utf-8 -*-

"""QJansenWidget.py: GUI for holographic video microscopy."""

from pyqtgraph.Qt import QtGui, QtCore
from .QJansenScreen import QJansenScreen
from .QHistogramTab import QHistogramTab
from .QDVRWidget import QDVRWidget
from common.tabLayout import tabLayout
from .video import QCameraPropertyWidget, QVideoFilterWidget
from tasks.taskmanager import taskmanager
from help.QHelpBrowser import QHelpBrowser
from .video.QDetector import QDetector


class QJansenWidget(QtGui.QWidget):

    def __init__(self, **kwargs):
        super(QJansenWidget, self).__init__(**kwargs)
        self.init_components()
        self.init_ui()

    def init_components(self):
        # video screen
        self.screen = QJansenScreen(gray=True, parent=self)
        self.wcamera = QCameraPropertyWidget(self.screen.video.source)
        self.detector = QDetector(parent=self.screen)
        self.filters = QVideoFilterWidget(self.screen.video)

        # tasks are processes that are synchronized with video frames
        self.tasks = taskmanager(parent=self)
        # DVR
        self.dvr = QDVRWidget(screen=self.screen)
        self.dvr.recording.connect(self.handleRecording)

    def handleRecording(self, recording):
        self.wcamera.enabled = not recording

    def init_ui(self):
        layout = QtGui.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.screen)
        self.tabs = QtGui.QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        index = self.tabs.addTab(self.videoTab(), 'Video')
        self.tabs.setTabToolTip(index, 'Video')
        tab = QHistogramTab(self.screen.video)
        tab.index = self.tabs.addTab(tab, 'Histogram')
        self.tabs.setTabToolTip(tab.index, 'Histogram')
        self.tabs.currentChanged.connect(tab.expose)
        self.tabs.addTab(self.helpTab(), 'Help')
        layout.addWidget(self.tabs)
        layout.setAlignment(self.tabs, QtCore.Qt.AlignTop)

    def setSize(self):
        """Fix the size of the tab bar

        Tabs should take up than 7 inches of screen space,
        and certainly less than a third of the screen"""
        desktop = QtGui.QDesktopWidget()
        rect = desktop.screenGeometry(0)
        width = min(rect.width() // 3, int(7*desktop.logicalDpiX()))
        self.tabs.setMaximumWidth(width)
        self.tabs.setFixedWidth(self.tabs.width())
        self.setMinimumWidth(self.screen.width()+self.tabs.width())
        return self

    def videoTab(self):
        wvideo = QtGui.QWidget()
        layout = tabLayout(wvideo)
        layout.addWidget(self.dvr)
        layout.addWidget(self.wcamera)
        layout.addWidget(self.filters)
        return wvideo

    def helpTab(self):
        whelp = QtGui.QWidget()
        self.browser = QHelpBrowser('jansen')
        bback = QtGui.QPushButton('Back')
        bback.clicked.connect(self.browser.back)
        layout = tabLayout(whelp)
        layout.addWidget(bback)
        layout.addWidget(self.browser)
        return whelp

    def keyPressEvent(self, event):
        key = event.text()
        if key == 'r':
            if self.dvr.is_recording():
                self.dvr.bstop.animateClick(100)
            else:
                self.dvr.brecord.animateClick(100)
        elif key == 's':
            self.dvr.bstop.animateClick(100)
        elif key == 'R':
            self.dvr.bstop.animateClick(100)
            self.dvr.getFilename()
            self.dvr.brecord.animateClick(100)
        event.accept()

    def close(self):
        self.screen.close()

    def closeEvent(self, evevnt):
        self.close()
