# -*- coding: utf-8 -*-

"""QJansenWidget.py: GUI for holographic video microscopy."""

from PyQt5.QtCore import (Qt, pyqtSlot)
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QPushButton,
                             QTabWidget, QDesktopWidget)
from .QJansenScreen import QJansenScreen
from .QHistogramTab import QHistogramTab
from .QDVR.QDVR import QDVR
from common.tabLayout import tabLayout
from .video.QVideoFilter.QVideoFilter import QVideoFilter
from tasks.taskmanager import Taskmanager
from help.QHelpBrowser import QHelpBrowser


class QJansenWidget(QWidget):

    def __init__(self, parent=None, camera=None):
        super(QJansenWidget, self).__init__(parent)
        # video screen
        self.screen = QJansenScreen(self, camera=camera)
        self.filters = QVideoFilter(self)

        # tasks are processes that are synchronized with video frames
        self.tasks = Taskmanager(self)

        # DVR
        self.dvr = QDVR(self)
        self.dvr.source = self.screen.source
        self.dvr.screen = self.screen
        self.dvr.recording.connect(self.handleRecording)

        self.init_ui()

    @pyqtSlot(bool)
    def handleRecording(self, recording):
        self.screen.camera.enabled = not recording

    def init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.screen)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        index = self.tabs.addTab(self.videoTab(), 'Video')
        self.tabs.setTabToolTip(index, 'Video')
        tab = QHistogramTab(self.screen)
        tab.index = self.tabs.addTab(tab, 'Histogram')
        self.tabs.setTabToolTip(tab.index, 'Histogram')
        self.tabs.currentChanged.connect(tab.expose)
        self.tabs.addTab(self.helpTab(), 'Help')
        layout.addWidget(self.tabs)
        layout.setAlignment(self.tabs, Qt.AlignTop)

    def setTabBarWidth(self):
        """Fix the width of the tab bar

        Tabs should take up than 7 inches of screen space,
        and certainly less than a third of the screen"""
        desktop = QDesktopWidget()
        rect = desktop.screenGeometry(0)
        width = min(rect.width() // 3, int(7 * desktop.logicalDpiX()))
        self.tabs.setMaximumWidth(width)
        self.tabs.setFixedWidth(self.tabs.width())
        vwidth = self.screen.camera.device.shape[1]
        width = vwidth + self.tabs.width()
        self.setMinimumWidth(min(width, rect.width()))
        self.setMaximumWidth(rect.width())
        return self

    def videoTab(self):
        wvideo = QWidget()
        layout = tabLayout(wvideo)
        layout.addWidget(self.dvr)
        layout.addWidget(self.screen.camera)
        layout.addWidget(self.filters)
        return wvideo

    def helpTab(self):
        whelp = QWidget()
        self.browser = QHelpBrowser('jansen')
        bback = QPushButton('Back')
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
