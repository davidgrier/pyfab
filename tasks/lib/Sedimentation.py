# -*- coding: utf-8 -*-
# MENU: Experiments/Sedimentation

from ..QTask import QTask
# from PyQt5.QtGui import QInputDialog


class Sedimentation(QTask):
    """Record sedimentation drop from a specified height."""

    def __init__(self, z=-200, **kwargs):
        super(Sedimentation, self).__init__(**kwargs)
        self.z = z
        #### GUI makes prompt obsolete
#         qtext, ok = QInputDialog.getText(self.parent,
#                                          'Sedimentation',
#                                          'z = ')
#         if ok:
#             self.z = float(qtext)
#         else:
#             self.z = None

    def initialize(self, frame):
        self.register('MoveToPlane', z=self.z)
#         self.register('Wait')
        self.register('Record', stop=False)
        self.register('ClearTraps')
