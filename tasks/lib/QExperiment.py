# -*- coding: utf-8 -*-
from ..QTask import QTask
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class QExperiment(QTask):
    
    def __init__(self, info=None, loop=1, **kwargs):
        super(QExperiment, self).__init__(**kwargs)
        self.info = info
        self.loop = loop
        self.name = 'QExperiment'
        
    def initialize(self, frame):
        if isinstance(self.info, str):
            with open('tasks/experiments/{}'.format(self.info), 'rb') as f:
                self.info = json.load(f)
        
        self.tasks = []           
        for settings in self.info:                     
            task = self.register(settings['name'])
            task.loop = self.loop
            task.__dict__.update(settings)
            task.widget.updateUi()
            self.tasks.append(task)
         
    def complete(self):
        if self.loop > 1:  
            Next = self.register(self.name, info=self.info, loop=self.loop-1)
    
        
        
    
    