# -*- coding: utf-8 -*-

from collections import deque
import importlib


class taskmanager(object):
    """taskmanager creates and manages a queue of task() objects
    for the pyfab/jansen system.

    Tasks are added to the queue with registerTask() and are
    performed on a first-come-first-served basis.
    Video frames are passed to the active task by handleTask().
    Once the active task is complete, it is cleaned up and replaced
    with the next task from the queue.
    """

    def __init__(self, parent):
        self.parent = parent
        self.source = parent.screen.video
        self.task = None
        self.queue = deque()

    def handleTask(self, frame):
        """Activates the next task in the queue, processes the
        next video frame, then cleans up the task if it is done."""
        if self.task is None:
            try:
                self.task = self.queue.popleft()
            except IndexError:
                self.source.sigNewFrame.disconnect(self.handleTask)
                return
            self.task.initialize(frame)
        self.task.process(frame)
        if self.task.isDone():
            self.task = None

    def registerTask(self, task, **kwargs):
        """Places the named task into the task queue."""
        if isinstance(task, str):
            try:
                taskmodule = importlib.import_module('tasks.' + task)
                taskclass = getattr(taskmodule, task)
                task = taskclass(**kwargs)
            except ImportError:
                print('could not import ' + task)
                return
        task.setParent(self.parent)
        self.queue.append(task)
        if self.task is None:
            self.source.sigNewFrame.connect(self.handleTask)
