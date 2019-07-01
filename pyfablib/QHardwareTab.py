# -*- coding: utf-8 -*-

"""Control panel for holographic trapping hardware"""

from PyQt5.QtWidgets import QWidget
from .proscan.QProscan import QProscan
from .IPG.QIPGLaser import QIPGLaser
from common.tabLayout import tabLayout

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class QHardwareTab(QWidget):

    def __init__(self, parent=None):
        super(QHardwareTab, self).__init__(parent)
        self.title = 'Hardware'
        self.index = -1
        self._has_content = False

        layout = tabLayout(self)
        try:
            self.wstage = QProscan()
            layout.addWidget(self.wstage)
            self._has_content = True
        except ValueError as ex:
            self.wstage = None
            logger.warning('Could not install stage: {}'.format(ex))
        try:
            self.wlaser = QIPGLaser()
            layout.addWidget(self.wlaser)
            self._has_content = True
        except ValueError as ex:
            self.wlaser = None
            logger.warning('Could not install laser: {}'.format(ex))

    def expose(self, index):
        if index == self.index:
            if self.wstage is not None:
                self.wstage.start()
            if self.wlaser is not None:
                self.wlaser.start()
        else:
            if self.wstage is not None:
                self.wstage.stop()
            if self.wlaser is not None:
                self.wlaser.stop()

    def has_content(self):
        return self._has_content
