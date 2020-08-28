# -*- coding: utf-8 -*-
from ..QTask import QTask
import json

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class QExperiment(QTask):
    
    def __init__(self, info=None, loop=None, **kwargs):
        super(QExperiment, self).__init__(**kwargs)
        self.info = info
        self.loop = loop
                                        
    def initialize(self, frame):
        if isinstance(self.info, str):
            with open('tasks/experiments/'+self.info, 'rb') as f:
                self.info = json.load(f)
                    
        for settings in self.info:                     
            task = self.register(settings['name'])
            task.loop = self.loop
            task.__dict__.update(settings)
            # task.loop = self.loop
            task.widget.updateUi()
         
    def complete(self):
        if self.loop > 1:  
            Next = self.register('QExperiment', info=self.info, loop=self.loop-1)
    
    