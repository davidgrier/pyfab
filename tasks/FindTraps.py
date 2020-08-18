from .MaxTask import MaxTask
import trackpy as tp


class FindTraps(MaxTask):

    def __init__(self, ntraps=None, **kwargs):
        super(FindTraps, self).__init__(**kwargs)
        self.ntraps = ntraps
        self.traps = None

    def dotask(self):
        self.traps = tp.locate(self.frame, 11,
                               characterize=False,
                               topn=self.ntraps)
        print(self.traps)
