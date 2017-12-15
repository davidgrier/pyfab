class taskmanager(object):

    def __init__(self, parent):
        self.parent = parent
        self.source = parent.fabscreen.video
        self.source.sigNewFrame.connect(self.handleTask)
        self.task = None

    def handleTask(self, frame):
        if self.task is None:
            return
        if self.task.isDone():
            del task
            self.task = None
        else:
            self.task.process(frame)
            
    def registerTask(self, task):
        if self.task is None:
            self.task = task
            self.task.parent = self.parent
