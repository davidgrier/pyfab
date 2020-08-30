# -*- coding: utf-8 -*-
# MENU: Auto-Trap


from ..QTask import QTask
from PyQt5.QtGui import QVector3D
import numpy as np

from pylorenzmie.analysis import Video

class AutoTrap(QTask):
    """Detect and trap particles on the screen."""

    def __init__(self, **kwargs):
        super(AutoTrap, self).__init__(**kwargs)
        self.source = 'post'
        
    # def initialize(self, frame): print('Error: Vision not implemented')
    def initialize(self, frames):
        video = Video(frames=frames)
        video.set_trajectories()
        trajs = video.trajectories
        for particle in set(trajs.particle.to_list()):
            df = trajs[trajs.particle==particle]
            self.parent().pattern.createTrap(r=QVector3D(np.mean(df.x_p), np.mean(df.y_p), np.mean(df.z_p)))
