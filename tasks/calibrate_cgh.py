from .maxtask import maxtask
from PyQt5.QtGui import QVector3D


class calibrate_cgh(maxtask):

    def __init__(self, **kwargs):
        super(calibrate_cgh, self).__init__(**kwargs)

    def initialize(self, frame):
        self.parent.pattern.clearTraps()
        rc = self.parent.cgh.rc
        r1 = rc + QVector3D(100, 0, 0)
        r2 = rc - QVector3D(0, 100, 0)
        r = [r1, r2]
        self.parent.pattern.createTraps(r)
