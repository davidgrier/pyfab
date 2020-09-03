# -*- coding: utf-8 -*-
# MENU: Motion/Translate

from .Assemble import Assemble

import numpy as np
import json

from PyQt5.QtGui import QVector3D

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class Translate(Assemble):
    
    def __init__(self, smooth=True, stepSize=1.0, dr=(0., 0., 0.), **kwargs):
        super(Translate, self).__init__(smooth=smooth, stepSize=stepSize, **kwargs)
        self.dx, self.dy, self.dz = dr
    
    def initialize(self, frame):
        if self.smooth:
            super(Translate, self).initialize(frame)
        else:
            self.nframes = 0
            try:
                if isinstance(self.traps, list):
                    for trap in self.traps:
                        trap.moveBy(QVector3D(self.dx, self.dy, self.dz))
                else:
                    self.traps.moveBy(QVector3D(self.dx, self.dy, self.dz))    
            except AttributeError:
                logger.warn('error: self.traps must be QTrapGroup, QTrap, or list of QTraps')
                

    def parameterize(self, traps):
        trajs = []
        dr = [self.dx, self.dy, self.dz]
        for trap in traps:
            r0 = [trap.r.x(), trap.r.y(), trap.r.z()]
            rf = [r0[0] + self.dx, r0[1] + self.dy, r0[2] + self.dz]
            trajs.append(np.vstack([r0, rf]))
            print(trajs)
        self.trajectories = trajs
                    

            
