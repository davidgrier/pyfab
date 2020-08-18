# -*- coding: utf-8 -*-
# MENU: GlobalCounter


"""
A* graph search for moving a set of traps to a set of targets


To subclass this method, just override aim() with a method which returns targets (analagous to overriding 'parameterize' in MoveTraps).
Any parameters can be passed as kwargs on __init__ and output must be a dict with keys QTrap and values tuple (x, y, z)
"""


from ..QTask import QTask
import numpy as np
from time import sleep, time
from PyQt5.QtCore import pyqtProperty

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GlobalCounter(QTask):

    def __init__(self, nframes=30, **kwargs):
        super(GlobalCounter, self).__init__(nframes=nframes, **kwargs)
#         self._blocking = False
        self.lag = 0.
        self.globalframe=0;
    
    def initialize(self, frame):
        sleep(self.lag)
        print('start!')
        
    def process(self, frame):
#         self.myframes.append(Frame(image=frame, framenumber=self._frame))
        print('frame number {}'.format(self.globalframe))
        self.globalframe += 1
#          sleep(0.5)
              
    def complete(self):
        print('Done counting')
        self.setData({'globalframe': self.globalframe})
        print('set taskData to {}'.format(self.data()))
              