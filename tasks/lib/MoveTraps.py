# -*- coding: utf-8 -*-

from ..QTask import QTask


class MoveTraps(QTask):
    '''Move specified traps along specified trajectory'''

    def __init__(self, traps=None, trajectory=None, **kwargs):
        super(MoveTraps, self).__init__(**kwargs)
        self.traps = traps
        self.trajectory = trajectory
        self.nframes = len(self.trajectory) * self.skip
        self.pattern = self.parent().pattern.pattern
        if isinstance(self.traps, list):
            self.process = self.processTraps
        else:
            self.process = self.processTrap

    def processTrap(self, frame):
        pos = self.trajectory.pop(0)
        self.traps.moveTo(pos)

    def processTraps(self, frame):
        positions = self.trajectory.pop(0)
        self.pattern.blockRefresh(True)
        map(lambda trap, pos: trap.moveTo(pos), self.traps, positions)
        self.pattern.blockRefresh(False)
        self.pattern.refresh()
