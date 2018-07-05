# -*- coding: utf-8 -*-

"""Framework for moving all current traps along some trajectory"""

from .task import task


class trek(task):

    def __init__(self, **kwargs):
        super(trek, self).__init__(**kwargs)
        self.traps = None

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        self.trajectories = self.parameterize(self.traps)

    def dotask(self):
        if self.traps.count() > 0:
            if self.trajectories is not None:
                # All paths must be same length
                N = list(self.trajectories.values())[0].shape[0]
                # Move along paths
                self.traps.select(True)
                for n in range(N):
                    self.register('delay', delay=1)
                    for trap in self.trajectories:
                        trajectory = self.trajectories[trap]
                        self.register('step', trap=trap, r=trajectory[n])

    def parameterize(self, traps, destinations=None):
        """
        Returns a dictionary of traps corresponding to their
        respective parameterization.

        Args:
            traps: QTrapGroup of all traps on the QTrappingPattern
        Keywords:
            destinations: list of (x, y, z) destinations for each trap
        """
        return None
