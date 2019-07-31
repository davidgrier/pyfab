# -*- coding: utf-8 -*-

'''Changes properties of ultra traps.'''

from .task import task
from pyqtgraph.Qt import QtGui
from pyfablib.traps.QUltraTrap import QUltraTrap


class modify(task):

    def __init__(self, group=None, NewDeltaZ=0, NewDeltaPi=0, **kwargs):
        super(modify, self).__init__(**kwargs)
        self.group = group; self.traps=group.flatten();
        self.NewDeltaZ = NewDeltaZ; self.NewDeltaPi = NewDeltaPi;

    def initialize(self, frame):
        for trap in self.traps:
            if isinstance(trap, QUltraTrap): 
            	trap.deltaZ=self.NewDeltaZ; trap.deltaPi=self.NewDeltaPi;
