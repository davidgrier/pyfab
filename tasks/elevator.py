# -*- coding: utf-8 -*-
# MENU: Experiments/Measure z continuous

from .task import task


class elevator(task):
    """Automatically move traps up then down in the z direction."""

    def __init__(self, **kwargs):
        super(elevator, self).__init__(**kwargs)

    def initialize(self, frame):
        self.register('movetocoordinate', z=200, correct=False)
        self.register('movetocoordinate')
