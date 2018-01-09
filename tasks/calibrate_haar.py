from maxtask import maxtask
import trackpy as tp


class calibrate_haar(maxtask):

    def __init__(self, **kwargs):
        super(calibrate_haar, self).__init__(**kwargs)

    def setParent(self, parent):
        self.parent = parent
        self.parent.pattern.clearTraps()
        self.trap = self.parent.pattern.createTrap([-100, 0])

    def dotask(self):
        f = tp.locate(self.frame, 11, topn=2, characterize=False)
        print(f)
