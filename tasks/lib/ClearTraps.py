# -*- coding: utf-8 -*-
# MENU: Clear traps

from ..QTask import QTask


class ClearTraps(QTask):
    """Delete all traps."""

    def __init__(self, **kwargs):
        super(ClearTraps, self).__init__(**kwargs)

    def complete(self):
        self.parent().pattern.clearTraps()
