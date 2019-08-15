# -*- coding: utf-8 -*-

'''Changes properties of ultra traps.'''

from .Task import Task
from pyfablib.traps.QUltraTrap import QUltraTrap


class Modify(Task):

    def __init__(self, group=None, NewDeltaZ=0, NewDeltaPhi=0, **kwargs):
        super(Modify, self).__init__(**kwargs)
        self.group = group
        self.traps = group.flatten()
        self.NewDeltaZ = NewDeltaZ
        self.NewDeltaPhi = NewDeltaPhi

    def initialize(self, frame):
        for trap in self.traps:
            if isinstance(trap, QUltraTrap):
                trap.deltaZ = self.NewDeltaZ
                trap.deltaPhi = self.NewDeltaPhi
