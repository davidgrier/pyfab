# -*- coding: utf-8 -*-
# MENU: Add trap/Corral

from .Task import Task
import numpy as np
from numpy.random import normal
from PyQt5.QtGui import QVector3D


class Corral(Task):
    """Project a ring of traps"""

    def __init__(self,
                 radius=250,
                 ntraps=40,
                 zoffset=80,
                 fuzz=1,
                 **kwargs):
        super(Corral, self).__init__(**kwargs)
        self.radius = radius
        self.ntraps = ntraps
        self.zoffset = zoffset
        self.fuzz = fuzz
        self.traps = None

    def dotask(self):
        sz = self.parent.screen.video.camera.size
        theta = np.linspace(0, 2.*np.pi, self.ntraps, endpoint=False)
        x = self.radius * np.cos(theta) + \
            normal(scale=self.fuzz, size=theta.size) + \
            sz.width()/2.
        y = self.radius * np.sin(theta) + \
            normal(scale=self.fuzz, size=theta.size) + \
            sz.height()/2.
        p = list(map(lambda x, y: QVector3D(x, y, self.zoffset), x, y))
        self.traps = self.parent.pattern.createTraps(p)
