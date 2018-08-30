# -*- coding: utf-8 -*-
# MENU: Add trap/Corral

from .task import task
import numpy as np
from numpy.random import normal
from PyQt4.QtGui import QVector3D


class corral(task):
    """Project a ring of traps"""

    def __init__(self,
                 radius=200,
                 ntraps=40,
                 zoffset=40,
                 fuzz=1,
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
        x = self.radius * np.cos(theta) + \
            normal(scale=self.fuzz, size=theta.size) + \
            sz.width()/2.
        y = self.radius * np.sin(theta) + \
            normal(scale=self.fuzz, size=theta.size) + \
            sz.height()/2.
        p = list(map(lambda x, y: QVector3D(x, y, self.zoffset), x, y))
        self.traps = self.parent.pattern.createTraps(p)
