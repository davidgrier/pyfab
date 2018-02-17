from pyqtgraph.Qt import QtGui
from proscan.QProscan import QProscan
from IPG.QIPGLaser import QIPGLaser
from common.tabLayout import tabLayout
import logging


class QHardwareTab(QtGui.QWidget):

    def __init__(self):
        super(QHardwareTab, self).__init__()
        self.title = 'Hardware'
        self.index = -1

        layout = tabLayout(self)
        try:
            self.wstage = QProscan()
            layout.addWidget(self.wstage)
        except ValueError as ex:
            self.wstage = None
            logging.warning('Could not install stage: %s', ex)
        try:
            self.wlaser = QIPGLaser()
            layout.addWidget(self.wlaser)
        except ValueError as ex:
            self.wlaser = None
            logging.warning('Could not install laser: %s', ex)

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
                self.wstage.stop()
