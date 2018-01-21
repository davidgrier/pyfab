from task import task
from PyQt4.QtGui import QVector3D


class createtrap(task):

    def __init__(self, x=100, y=100, z=0, **kwargs):
        super(createtrap, self).__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z

    def initialize(self):
        print('createtrap')
        pos = QVector3D(self.x, self.y, self.z)
        self.parent.pattern.createTraps(pos)
        self.done = True
