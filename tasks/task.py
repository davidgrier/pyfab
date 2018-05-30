# -*- coding: utf-8 -*-
import copy

class task(object):
    """task is a base class for operations on images in pyfab/jansen

    Once a task() is registered with taskmanager().registerTask()
    the taskmanager queues it until it is ready to run.  At that point,
    taskmanager calls initialize() and proceeds to feed video frames
    to the task as they become available. The task performs
    doprocess() on a number of frames set by nframes
    (default: nframes = 0) while between each doprocess() the task skips a number
    of frames set by delay (default: delay = 0) before it finally performs its task
    with a call to dotask().

    When the task isDone(), jansen unregisters the task and deletes it.

    Subclasses of task() should override
    initialize(), doprocess() and dotask()
    as needed to to accomplish their goals.
    """

    def __init__(self,
                 parent=None,
                 delay=0,
                 nframes=0):
        self.setParent(parent)
        self.done = False
        self.delay = delay
        self.delayOriginal = delay
        self.nframes = nframes

    def isDone(self):
        return self.done

    def setParent(self, parent):
        self.parent = parent

    def initialize(self, frame):
        """Called when the taskmanager activates the task."""
        pass

    def doprocess(self, frame):
        """Operation performed on each video frame."""
        pass

    def dotask(self):
        """Operation performed to complete the task."""
        pass

    def process(self, frame):
        if self.delay > 0:
            self.delay -= 1
        elif self.nframes > 0:
            self.delay = copy.deepcopy(self.delayOriginal)
            self.doprocess(frame)
            self.nframes -= 1
        else:
            self.dotask()
            print('TASK: ' + self.__class__.__name__ + ' done')
            self.done = True
