# -*- coding: utf-8 -*-
# MENU: Task Data/Examples/Example 3


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

############ Before you run this task, make a trap or trapgroup and click on it!


class TaskDataExample3(QTask):
        
    def initialize(self, frame):
        print('Before you run this task, make a trap or trapgroup and click on it!')
        print("We can use task data to set task parameters. Let's run encircle a few times, and use task data to specify where the center is. We can also add to existing task parameters by using non-blocking tasks.")

        self.register('setTaskData', new_data={'xc': 100., 'yc': 100.})
        self.register('Encircle', nframes=100)
        self.register('setTaskData', new_data=['xc', 'yc'])
        self.register('Encircle', nframes=100)
              