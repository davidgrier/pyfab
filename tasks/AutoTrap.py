# -*- coding: utf-8 -*-
# MENU: Auto-Trap
# VISION: True

from .Task import Task
from PyQt5.QtGui import QVector3D


class AutoTrap(Task):
    """Detect and trap particles on the screen."""

    def __init__(self, **kwargs):
        super(AutoTrap, self).__init__(**kwargs)

    def initialize(self, frame):
        vision = self.parent.vision
        coords = []
        for feature in frame.features:
            z = None
            particle = feature.model.particle
            if vision.estimate:
                # TODO: Correct for discrepency btwn
                # focal plane & trapping plane
                correction = 0
                z = particle.z_p + correction
            elif vision.detect:
                z = 0
            if z is not None:
                x, y = (particle.x_p, particle.y_p)
                coord = QVector3D(x, y, z)
                coords.append(coord)
        self.parent.pattern.createTraps(coords)
