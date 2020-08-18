# -*- coding: utf-8 -*-
# MENU: Task Data/Set Task Data

from ..QTask import QTask
from PyQt5.QtWidgets import QInputDialog

    

class setTaskData(QTask):
    """Prompt user to set Task Data (or pass as kwarg) to pass to the next Task. (Only supports doubles)"""
    def prompt(self, str):
        qparam, ok = QInputDialog.getDouble(self.parent(), 'Parameters', str)
        if ok:
            return qparam
        else:
            return None

    def __init__(self, new_data=None, **kwargs):
        super(setTaskData, self).__init__(**kwargs)
#         self._blocking = False
        self.new_data = new_data
    
    def complete(self):
        if isinstance(self.new_data, dict):
            new_data = self.new_data
        elif self.new_data is None:
            new_data = {}
            
            ok = False
            while not ok:
                N, ok = QInputDialog.getInt(self.parent(), 'Parameters', 'How many items in task data? (int)')
            
            for i in range(N):
                ok = False
                while not ok:
                    label, ok = QInputDialog.getText(self.parent(), 'Parameters', 'Enter the name of item {}'.format(i))
                ok = False
                while not ok:
                    val, ok = QInputDialog.getDouble(self.parent(), 'Parameters', "Enter the values for key '{}'".format(label))
                new_data[label] = val
            
        elif isinstance(self.new_data, list) and all([isinstance(el, str) for el in self.new_data]):
            new_data = {}
            for label in self.new_data:
                ok = False
                while not ok:
                    val, ok = QInputDialog.getDouble(self.parent(), 'Parameters', "Enter the values for key '{}'".format(label))
                new_data[label] = val
        
        else:
#             print(self.new_data)
#             print(type(self.new_data))
            print('error: taskData is not a dict')
            return
        print(new_data)
        self.setData(new_data)
        
        