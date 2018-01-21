class task(object):

    def __init__(self,
                 parent=None,
                 delay=0,
                 nframes=0):
        self.setParent(parent)
        self.done = False
        self.delay = delay
        self.nframes = nframes

    def isDone(self):
        return self.done

    def setParent(self, parent):
        self.parent = parent

    def initialize(self):
        pass

    def doprocess(self, frame):
        pass

    def dotask(self):
        pass
        
    def process(self, frame):
        if self.delay > 0:
            self.delay -= 1
        elif self.nframes > 0:
            self.doprocess(frame)
            self.nframes -= 1
        else:
            self.dotask()
            print('TASK: ' + self.__class__.__name__ + ' done')
            self.done = True
