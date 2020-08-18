# -*- coding: utf-8 -*-
# MENU: Task Data/Examples/Example 1


"""
Examples of using Task Data
"""


from ..QTask import QTask


############ Before you run this task, make a trap or trapgroup and click on it!


class TaskDataExample1(QTask):
        
    def initialize(self, frame):
        print("1) Blocking tasks can set task data. Let's register two 'Global Counter' tasks; the property 'global_frame' is passed from the first task to the next.")
        print("More specifically, the property 'global_frame' is saved to taskData by the first counter on complete(), and used to set the property 'global_frame' of the second task after __init__()")
        self.register('GlobalCounter', blocking=True, nframes=10)
        self.register('GlobalCounter', blocking=True, nframes=10)
        print()
        print("2) If the next task in the queue does not have a property 'global_frame', then that property is removed from the taskData. Let's register 'OpticalTweezer' and then register 'Global Counter' again.") 
        self.register('OpticalTweezer')
        self.register('GlobalCounter', blocking=True, nframes=10)
      
