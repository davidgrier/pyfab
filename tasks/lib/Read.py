# -*- coding: utf-8 -*-
# MENU: Experiments/Read

from .MoveTraps import Move
import numpy as np
import json


class Read(Move):
    """Make particles move in a circle around some point"""
    
    def __init__(self, **kwargs):
        with open('tasks/lib/trajectories.json', 'r') as f:
            data = json.load(f)
        
#         print('read trajectories {}'.format(list(data.values())))
        super(Read, self).__init__(trajectories=list(data.values()), **kwargs)
       
