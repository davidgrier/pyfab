from task import task


class stagego(task):

    def __init__(self,
                 dx = -100,
                 speed=1,
                 **kwargs):
        super(stagego, self).__init__(**kwargs)
        self.dx = dx
        self.speed = speed
        self.nframes = 10

    def initialize(self):
        self.wstage = self.parent.wstage
        self.stage = self.wstage.instrument
        self.position = self.stage.position()
        self.stage.setMaxSpeed(self.speed)
        self.goal = self.position[0] + self.dx
        self.stage.moveX(self.goal)

    def doprocess(self, frame):
        if self.stage.stageMoving():
            self.nframes = 2

    def dotask(self):
        self.stage.reset()
