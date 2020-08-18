# -*- coding: utf-8 -*-
# MENU: Experiments/Volumetric Image

from .Task import Task
from PyQt5.QtGui import QVector3D
import os

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class VolumetricImage(Task):
    """Do a max task, displace trap in z, and repeat"""

    def __init__(self, **kwargs):
        super(VolumetricImage, self).__init__(**kwargs)

    def initialize(self, frame):
        # Set traps from last QTrapGroup created
        pattern = self.parent.pattern.pattern
        group = None
        for child in reversed(pattern.children()):
            if isinstance(child, type(pattern)):
                group = child
                break
        self.group = group

    def dotask(self):
        traps = self.group.flatten()
        if len(traps) == 1:
            trap = traps[0]
            fn, fn_ext = os.path.splitext(self.parent.dvr.filename)
            zmax = -100
            z = trap.r.z()
            dz = -3
            dr = QVector3D(0, 0, dz)
            while z >= zmax:
                prefix = fn+'_'+str(z)
                self.register('MaxTask', prefix=prefix)
                self.register('Translate', traps=trap, dr=dr)
                z += dz
        else:
            logger.warning("Please create a QTrapGroup with one trap.")
