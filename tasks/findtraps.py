from .maxtask import maxtask
import trackpy as tp


class findtraps(maxtask):

    def __init__(self, ntraps=None, **kwargs):
        super(findtraps, self).__init__(**kwargs)
        self.ntraps = ntraps
        self.traps = None

    def dotask(self):
        self.traps = tp.locate(self.frame, 11,
                               characterize=False,
                               topn=self.ntraps)
        print(self.traps)
