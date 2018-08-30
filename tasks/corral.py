# -*- coding: utf-8 -*-
# MENU: Add trap/Corral

import numpy as np
from numpy.random import normal
from .task import task
from PyQt4.QtGui import QVector3D
import os


class corral(task):
    """Project a ring of traps"""

    def __init__(self,
                 radius=200,
                 ntraps=40,
                 zoffset=40,
                 fuzz=0.02,
                 **kwargs):
        super(corral, self).__init__(**kwargs)
        self.radius = radius
        self.ntraps = ntraps
        self.zoffset = zoffset
        self.fuzz = fuzz
        self.traps = None

    def dotask(self):
        sz = self.parent.screen.video.camera.size
        theta = np.linspace(0, 2.*np.pi, self.ntraps, endpoint=False)
        x = self.radius * np.cos(theta) + sz.width()/2. + \
            normal(scale=self.fuzz, size=theta.size)
        y = self.radius * np.sin(theta) + sz.height()/2. + \
            normal(scale=self.fuzz, size=theta.size)
        p = list(map(lambda x, y: QVector3D(x, y, self.zoffset), x, y))
        self.traps = self.parent.pattern.createTraps(p)
