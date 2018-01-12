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
            self.task.initialize()
        self.task.process(frame)
        if self.task.isDone():
            self.task = None

    def registerTask(self, task, **kwargs):
        if isinstance(task, str):
            try:
                taskmodule = importlib.import_module('tasks.'+task)
                taskclass = getattr(taskmodule, task)
                task = taskclass(**kwargs)
            except ImportError:
                print('could not import '+task)
                return
        task.setParent(self.parent)
        self.queue.append(task)
        if self.task is None:
            self.source.sigNewFrame.connect(self.handleTask)
