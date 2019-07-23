#!/usr/bin/env python
# -*- coding: utf-8 -*-

from common.QSettingsWidget import QSettingsWidget
from .QCGHWidget import Ui_QCGHWidget


import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QCGH(QSettingsWidget):

    '''Widget for setting CGH calibration constants'''

    def __init__(self, parent=None, device=None):
        ui = Ui_QCGHWidget()
        super(QCGH, self).__init__(parent=parent,
                                   device=device,
                                   ui=ui)


if __name__ == '__main__':
    from PyQt5.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    wid = QCGH()
    wid.show()
    sys.exit(app.exec_())
