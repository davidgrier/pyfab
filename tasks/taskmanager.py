from collections import deque
import importlib


class taskmanager(object):

    def __init__(self, parent):
        self.parent = parent
        self.source = parent.fabscreen.video
        self.task = None
        self.queue = deque()

    def handleTask(self, frame):
        if self.task is None:
            try:
                self.task = self.queue.popleft()
            except IndexError:
                self.source.sigNewFrame.disconnect(self.handleTask)
                return
        self.task.process(frame)
        if self.task.isDone():
            self.task = None

    def registerTask(self, task):
        if isinstance(task, str):
            try:
                taskmodule = importlib.import_module('tasks.'+task)
                taskclass = getattr(taskmodule, task)
                task = taskclass()
            except ImportError:
                print('could not import '+task)
                return
        if self.task is None:
            self.task = task
            self.task.parent = self.parent
            self.source.sigNewFrame.connect(self.handleTask)
        else:
            self.queue.append(task)
