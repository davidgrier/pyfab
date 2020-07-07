# -*- coding: utf-8 -*-

from ..QTask import QTask


class MoveTraps(QTask):
    '''Move specified traps along specified trajectory'''

    def __init__(self, traps=None, trajectories=None, **kwargs):
        super(MoveTraps, self).__init__(**kwargs)
        self.traps = traps if isinstance(traps, list) else [traps]
        self.trajectories = trajectories if isinstance(trajectories, list) else [trajectories]
        self.nframes = len(self.trajectories[0]) * self.skip
        self.pattern = self.parent().pattern.pattern

    def process(self, frame):
        positions = [traj.pop(0) for traj in self.trajectories]
        self.pattern.blockRefresh(True)
        map(lambda trap, pos: trap.moveTo(pos), self.traps, positions)
        self.pattern.blockRefresh(False)
        self.pattern.refresh()

        
