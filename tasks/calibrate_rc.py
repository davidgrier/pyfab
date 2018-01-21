from maxtask import maxtask
import trackpy as tp


class calibrate_rc(maxtask):

    def __init__(self, **kwargs):
        super(calibrate_rc, self).__init__(**kwargs)

    def initialize(self):
        self.parent.pattern.clearTraps()

    def dotask(self):
        f = tp.locate(self.frame, 11, topn=1, characterize=False)
        self.parent.wcgh.xc = f['x']
        self.parent.wcgh.yc = f['y']
