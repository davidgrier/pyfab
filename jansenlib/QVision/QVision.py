# -*- coding: utf-8 -*-

from PyQt5.QtWidgets import QWidget
from .QVisionWidget import Ui_QVisionWidget

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QVision(QWidget):

    def __init__(self, parent=None, nskip=3):
        super(QVision, self).__init__(parent)
        self.ui = Ui_QVisionWidget()
        self.ui.setupUi(self)
