# -*- coding: utf-8 -*-

"""QJansenWidget.py: GUI for holographic video microscopy."""

from PyQt5 import QtGui, QtCore
from QJansenScreen import QJansenScreen
from QHistogramTab import QHistogramTab
from QDVRWidget import QDVRWidget
from common.tabLayout import tabLayout
from .video import QVideoFilterWidget
from .video.QOpenCV.QOpenCV import QOpenCV
from tasks.taskmanager import Taskmanager
from help.QHelpBrowser import QHelpBrowser


class QJansenWidget(QtGui.QWidget):

    def __init__(self, parent=None):
        super(QJansenWidget, self).__init__(parent)
        self.init_components()
        self.init_ui()

    def init_components(self):
        # video source
        self.wcamera = QOpenCV()

        # video screen
        self.screen = QJansenScreen(self, camera=self.wcamera)
        self.filters = QVideoFilterWidget(self)

        # tasks are processes that are synchronized with video frames
        self.tasks = Taskmanager(self)

        # DVR
        self.dvr = QDVRWidget(self)
        self.dvr.recording.connect(self.handleRecording)

    @QtCore.pyqtSlot(bool)
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

    def setTabBarWidth(self):
        """Fix the width of the tab bar

        Tabs should take up than 7 inches of screen space,
        and certainly less than a third of the screen"""
        desktop = QtGui.QDesktopWidget()
        rect = desktop.screenGeometry(0)
        width = min(rect.width() // 3, int(7 * desktop.logicalDpiX()))
        self.tabs.setMaximumWidth(width)
        self.tabs.setFixedWidth(self.tabs.width())
        width = self.screen.video.camera.width()
        self.setMinimumWidth(width + self.tabs.width())
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
