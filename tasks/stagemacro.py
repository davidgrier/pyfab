from task import task


class stagemacro(task):

    def __init__(self,
                 macro = None,
                 **kwargs):
        super(stagemacro, self).__init__(**kwargs)
        self.macro = ['SOAK',
                      'GR,100,0,0',
                      'WAIT,100',
                      'GR,-100,0,0',
                      'WAIT,100',
                      'SOAK']
        self.target = 5

    def initialize(self):
        self.wstage = self.parent.wstage
        self.stage = self.wstage.instrument
        self.target = self.stage.position()
        for cmd in self.macro:
            self.stage.command(cmd)
        self.nframes = 1000

    def doprocess(self, frame):
        if self.stage.available():
            print(self.stage.readln())
            self.nframes = 2
