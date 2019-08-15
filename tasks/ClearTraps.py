# -*- coding: utf-8 -*-
# MENU: Clear traps

from .Task import Task


class ClearTraps(Task):
    """Delete all traps."""

    def __init__(self, **kwargs):
        super(ClearTraps, self).__init__(**kwargs)

    def dotask(self):
        self.parent.pattern.clearTraps()
