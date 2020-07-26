# -*- coding: utf-8 -*-
# MENU: Experiments/Read2

from ..QTask import QTask
import numpy as np
import json


class Read2(QTask):
    """Make particles move in a circle around some point"""
    
    def initialize(self, frame):
        with open('tasks/lib/trajectories.json', 'r') as f:
            data = json.load(f)
        trajectories = list(data.values())
        targets = [traj[0] for traj in trajectories]
        self.register('AssembleTraps', targets=targets)
        self.register('MoveTraps', trajectories=trajectories)
            
        