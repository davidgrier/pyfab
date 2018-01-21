from task import task


class stagesinusoid(task):

    def __init__(self,
                 amplitude=10.,
                 speed=10.,
                 cycles=1,
                 **kwargs):
        super(stagesinusoid, self).__init__(**kwargs)
        self.stage = self.parent.wstage.instrument
        self.position = self.stage.position()
        self.amplitude = amplitude
        self.cycles = cycles
        acceleration = speed**2/(6.*amplitude)
        scurve = acceleration**2/(2.*speed)
        self.stage.setMaxSpeed(1.5*speed)
        self.stage.setAcceleration(acceleration)
        self.stage.setSCurve(scurve)

    def initialize(self):
        x0 = self.position[0]
        self.goals = [x0 + self.amplitude]
        for n in range(self.cycles):
            self.goals.extend([x0 - self.amplitude,
                               x0 + self.amplitude])
        self.goals.append(x0)
        self.stage.moveX(self.goals[0])

    def doprocess(self, frame):
        if self.stage.x() == self.goals[0]:
            self.goals.pop(0)
            if len(self.goals) > 0:
                self.stage.moveX(self.goals[0])
        self.nframes = len(self.goals)

    def dotask(self):
        self.stage.reset()
