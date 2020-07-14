# -*- coding: utf-8 -*-

from ..QTask import QTask


class MoveTraps(QTask):
    '''Move specified traps along specified trajectory'''

    def __init__(self, traps=None, trajectory=None, **kwargs):
        super(MoveTraps, self).__init__(**kwargs)
        self.pattern = self.parent().pattern.traps
        self.traps = traps if isInstance(traps, list) else [traps]
        self.trajectories = trajectory if isInstance(trajectories, list) else [trajectories]
        self.nframes = len(self.trajectory) * self.skip

    def process(self, frame):
        positions = [traj.pop(0) for traj in self.trajectories]
        self.pattern.blockRefresh(True)
        map(lambda trap, pos: trap.moveTo(pos), self.traps, positions)
        self.pattern.blockRefresh(False)
        self.pattern.refresh()
