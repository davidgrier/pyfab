# -*- coding: utf-8 -*-
# MENU: Experiments/Volumetric Image

from .Task import Task
from PyQt5.QtGui import QVector3D
from pyfablib.traps.QCustomTrap import QCustomTrap
import os


class VolumetricImage(Task):
    """Do a max task, displace trap in z, and repeat"""

    def __init__(self, **kwargs):
        super(VolumetricImage, self).__init__(**kwargs)

    def initialize(self, frame):
        traps = self.parent.pattern.pattern.flatten()
        for trap in traps:
            if isinstance(trap, QCustomTrap):
                self.trap = trap

    def dotask(self):
        fn, fn_ext = os.path.splitext(self.parent.dvr.filename)
        zmax = -100
        z = self.trap.r.z()
        dz = -3
        dr = QVector3D(0, 0, dz)
        while z >= zmax:
            prefix = fn+'_'+str(z)
            self.register('MaxTask', prefix=prefix)
            self.register('Translate', traps=self.trap, dr=dr)
            z += dz
