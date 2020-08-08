# -*- coding: utf-8 -*-
# MENU: Experiments/Repeat

from ..QTask import QTask
import numpy as np
import json


class Repeat(QTask):
    """Repeat the trajectory from the previous task"""
    
    def initialize(self, frame):
        with open('home/python/pyfab/tasks/data/trajectories.json', 'r') as f:
            data = json.load(f)
        trajectories = list(data.values())
        targets = [traj[0] for traj in trajectories]
        self.register('Assemble', targets=targets)
        self.register('Move', trajectories=trajectories)
