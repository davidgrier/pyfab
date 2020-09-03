# -*- coding: utf-8 -*-
# MENU: Auto-Trap


from ..QTask import QTask
from PyQt5.QtGui import QVector3D
import numpy as np

from pylorenzmie.analysis import Video

class AutoTrap(QTask):
    """Detect and trap particles on the screen."""

    def __init__(self, source='post', z_default=0., use_bboxes=False, **kwargs):
        super(AutoTrap, self).__init__(**kwargs)
        self.source = source
        self.z_default = z_default
        self.use_bboxes = use_bboxes
        
    def initialize(self, frames):
        if self.source == 'post':
            video = Video(frames=frames)
            video.set_trajectories()
            trajs = video.trajectories
            for particle in set(trajs.particle.to_list()):
                df = trajs[trajs.particle==particle]           
                self.parent().pattern.createTrap(r=QVector3D(np.mean(df.x_p), np.mean(df.y_p), np.mean(df.z_p)))    
    
    def process(self, frame):
        if self.source == 'realtime':
            self.parent().pattern.clearTraps()
            if self.use_bboxes:
                for bbox in frame.bboxes:
                    if bbox is not None:
                        self.parent().pattern.createTrap(r=QVector3D(bbox[0], bbox[1], self.z_default))
            else:
                for feat in frame.features:
                    self.parent().pattern.createTrap(r=QVector3D(feat.particle.x_p, feat.particle.y_p, 0))
            

                    
