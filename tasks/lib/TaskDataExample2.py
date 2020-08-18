# -*- coding: utf-8 -*-
# MENU: Task Data/Examples/Example 2


"""
Examples of using Task Data
"""


from ..QTask import QTask
import numpy as np
from time import time
from PyQt5.QtCore import pyqtProperty

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

############ Before you run this task, make a trap or trapgroup and click on it!


class TaskDataExample2(QTask):
        
    def initialize(self, frame):
        self.nframes = 60
        print("Non-blocking tasks can also get and set data, but they won't clear any unused properties.")
        print("    Counter 1 is blocking: it sets data[globalframe]=10 when it finishes.")
        print("    If we register OpticalTweezer as non-blocking, it doesn't change data[globalframe], so the next counter will still start with globalframe=10")
        print("    Let's register Counter 2 as non-blocking for a long time; 25 frames. Since Counter2 has a property 'globalframe', it removes it from the task data.")
        print("    If we register Counter 3 five frames later for 5 frames, it starts at globalframe=0 and then sets data[globalframe] = 5")
        print("    If we register another counter (Counter 4) after Counter 3, it will start with globalframe=5")
        print("    Finally, when Counter 2 finishes, it will set data[globalframe]=35, overwriting the value from Counter 4. If we start Counter 5 after Counter 2, then it will start with globalframe=35")
    
    def process(self, frame):
        print('Frame {}:'.format(self._frame))
        if self._frame is 0:  
            print('counter 1')
            self.register('GlobalCounter', blocking=False, nframes=10)            ## Counter 1
        elif self._frame is 15: 
            self.register('OpticalTweezer', blocking=False)
        elif self._frame is 20: 
            print('counter 2')
            self.register('GlobalCounter', blocking=False, nframes=25) ## Counter 2
        elif self._frame is 26: 
            print('counter 3')
            self.register('GlobalCounter', blocking=False, nframes=5)  ## Counter 3
        elif self._frame is 31: 
            print('counter 4')
            self.register('GlobalCounter', blocking=False, nframes=5)  ## Counter 4
        elif self._frame is 46: 
            print('counter 5')
            self.register('GlobalCounter', blocking=False, nframes=5)  ## Counter 5
    