# -*- coding: utf-8 -*-
# MENU: Experiments/Sedimentation

from .Task import Task
from PyQt5.QtGui import QInputDialog


class Sedimentation(Task):
    """Record sedimentation drop from a specified height."""

    def __init__(self, z=-200, **kwargs):
        super(Sedimentation, self).__init__(**kwargs)
        qtext, ok = QInputDialog.getText(self.parent,
                                         'Sedimentation',
                                         'z = ')
        if ok:
            self.z = float(qtext)
        else:
            self.z = None

    def initialize(self, frame):
        self.register('MoveToPlane', z=self.z)
        self.register('Wait')
        self.register('Record', stop=False)
        self.register('ClearTraps')
