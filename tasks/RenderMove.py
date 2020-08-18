# -*- coding: utf-8 -*-

from .RenderText import RenderText
from PyQt5.QtGui import QVector3D


class RenderMove(RenderText):
    """Demonstrates trap motion under programmatic control.

    Render the word hello, wait 30 frames, and then move the
    traps horizontally by 10 pixels."""

    def __init__(self):
        super(RenderMove, self).__init__()
        self.delay = 30

    def dotask(self):
        dr = QVector3D(10, 0, 0)
        self.traps.select(True)
        self.traps.moveBy(dr)
        self.traps.select(False)
