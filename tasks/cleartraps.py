from task import task

class cleartraps(task):

    def __init__(self, **kwargs):
        super(cleartraps, self).__init__(**kwargs)

    def initialize(self):
        print('cleartraps')
        self.parent.pattern.clearTraps()
        self.done = True
