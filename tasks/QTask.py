# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSlot, pyqtSignal, pyqtProperty)
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
    It then feeds nframes frames to process() (default: 0),
    skipping skip frames between operations (default: 1).
    Finally, the task calls complete() to complete its operation.

    When the task isDone(), taskmanager() unregisters the task
    and deletes it.

    Subclasses of QTask() should override
    initialize(), process() and complete()
    A subclass that allocates resources can free them by overriding
    shutdown().
    """

    sigDone = pyqtSignal()

    def __init__(self, delay=0, nframes=0, skip=1, blocking=True, **kwargs):
        super(QTask, self).__init__(**kwargs)
        self._blocking = blocking
        self.delay = delay
        self.nframes = nframes
        self.skip = skip
        self._initialized = False
        self._paused = False
        self._busy = False
        self._frame = 0
        self._data = dict()
        self.register = self.parent().tasks.registerTask

    def initialize(self, frame):
        """Perform initialization operations"""
        logger.debug('Initializing')

    def process(self, frame):
        """Operation performed on each video frame."""
        logger.debug('Processing')

    def complete(self):
        """Operation performed to complete the task."""
        logger.debug('Completing')

    def shutdown(self):
        """Clean up resources"""
        logger.debug('Cleaning up')

    def setData(self, data):
        self._data = data or dict()

    def data(self):
        return self._data

    @pyqtProperty(bool)
    def blocking(self):
        return self._blocking

    @pyqtSlot(np.ndarray)
    def handleTask(self, frame):
        logger.debug('Handling Task')
        try:
            self._handleTask(frame)
        except Exception ex:
            self.busy = True
            logger.warning('Killing task : {}'.format(ex))
            self.data['error'] = ex
            self.stop()
            
    def _handleTask(self, frame):
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
            self._frame += 1
            self._busy = False
            return
        self._busy = True
        self.complete()
        self.stop()

    @pyqtSlot(bool)
    def pause(self, state):
        self._paused = state

    @pyqtSlot()
    def stop(self):
        self.shutdown()
        self.sigDone.emit()
        logger.info('TASK: {} done'.format(self.__class__.__name__))
