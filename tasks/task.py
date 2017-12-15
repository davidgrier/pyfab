class task(object):

    def __init__(self):
        self.parent = None
        self.done = False

    def isDone(self):
        return self.done

    def process(self, frame):
        self.done = True
