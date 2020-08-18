# -*- coding: utf-8 -*-
# MENU: Motion/Move to trapping plane

from .Assemble import Assemble
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

#### This is basic enough that it might be worth removing, but is definitely a good example task
class MoveToPlane(Assemble):
    """Move traps to desired xy plane. By default moves to z = 0."""

    def __init__(self, z=0, **kwargs):
        super(MoveToPlane, self).__init__(**kwargs)
        self.z = z

    def aim(self, traps):
        vertices = []
        for trap in traps:
            vertices.append( (trap.r.x(), trap.r.y(), self.z) )
        self.targets = vertices