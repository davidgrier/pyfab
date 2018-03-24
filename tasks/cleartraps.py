# -*- coding: utf-8 -*-

from task import task


class cleartraps(task):
    """Delete all traps."""

    def __init__(self, **kwargs):
        super(cleartraps, self).__init__(**kwargs)

    def dotask(self):
        self.parent.pattern.clearTraps()
