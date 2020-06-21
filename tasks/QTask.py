# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSlot, pyqtSignal)
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARN)


class QTask(QObject):
    """QTask is a base class for operations on images in pyfab/jansen

    Registering a task with QTaskmanager().registerTask() places the
    task in a queue.  When the task reaches the head of the queue,
    QTaskmanager() connects the handleTask() slot to a signal
    that provides image data.

    The task skips a number of frames set by delay (default: 0).
    It then feeds a number of frames to doprocess() set by
    nframes (default: 0).
    Finally, the task calls dotask() to perform its operation.

    When the task isDone(), taskmanager() unregisters the task
    and deletes it.

    Subclasses of QTask() should override
    initialize(), process() and complete()
    """

    sigDone = pyqtSignal()

    def __init__(self, parent=None,
                 delay=0,
                 nframes=0,
                 skip=1):
        super(QTask, self).__init__(parent)
        self.skip = skip
        self.delay = delay
        self.nframes = nframes
        self._initialized = False
        self._paused = False
        self._busy = False
        self._frame = 0
        self.register = parent.tasks.registerTask

    def initialize(self, frame):
        """Perform initialization operations"""
        logger.debug('Initializing')

    def process(self, frame):
        """Operation performed on each video frame."""
        logger.debug('Processing')

    def complete(self):
        """Operation performed to complete the task."""
        logger.debug('Completing')

    @pyqtSlot(np.ndarray)
    def handleTask(self, frame):
        logger.debug('Handling Task')
        if not self._initialized:
            self.initialize(frame)
            self._initialized = True
        if (self._paused or self._busy):
            return
        if self.delay > 0:
            self.delay -= 1
            return
        if self._frame < self.nframes:
            if (self._frame % self.skip == 0):
                self._busy = True
                self.process(frame)
            self._frames += 1
            self._busy = False
        else:
            self._busy = True
            self.complete()
            self.sigDone.emit()
            logger.info('TASK: {} done'.format(self.__class__.__name__))

    @pyqtSlot(bool)
    def pause(self, state):
        self._paused = state
