# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSlot, pyqtSignal, pyqtProperty)
from .QTask import QTask
from collections import deque
import importlib

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QTaskmanager(QObject):

    """QTaskmanager creates and manages a queue of QTask() objects
    for the pyfab/jansen system.

    Tasks are added to the queue with registerTask() and are
    performed on a first-come-first-served basis.
    Video frames are passed to the active task by handleTask().
    Once the active task is complete, it is cleaned up and replaced
    with the next task from the queue.

    Non-blocking tasks are registered by setting blocking=False
    in the call to registerTask(). Such background tasks
    start running when registered and run in parallel with the
    task queue without blocking queued tasks.
    """

    sigPause = pyqtSignal(bool)
    sigStop = pyqtSignal()

    def __init__(self, parent=None):
        super(QTaskmanager, self).__init__(parent)
        self.source = self.parent().screen.source
        self.task = None
        self.tasks = deque()
        self.bgtasks = []
        self._paused = False

    def registerTask(self, taskname, blocking=True, **kwargs):
        """Places the named task into the task queue."""
        if isinstance(taskname, str):
            try:
                taskmodule = importlib.import_module('tasks.lib.' + taskname)
                taskclass = getattr(taskmodule, taskname)
                task = taskclass(parent=self.parent(),
                                 blocking=blocking, **kwargs)
            except ImportError as err:
                logger.error('Could not import {}: {}'.format(task, err))
                task = None
        self.queueTask(task)
        return task

    def connectSignals(self, task):
        task.sigDone.connect(lambda: self.dequeueTask(task))
        self.sigPause.connect(task.pause)
        self.sigStop.connect(task.stop)
        self.source.sigNewFrame.connect(task.handleTask)

    def disconnectSignals(self, task):
        try:
            self.source.sigNewFrame.disconnect(task.handleTask)
        except AttributeError:
            logger.warn('could not disconnect signals')

    def queueTask(self, task=None):
        """Add task to queue and activate next queued task if necessary"""
        if task:
            if task.blocking:
                self.tasks.append(task)
                logger.debug('Queuing blocking task')
            else:
                self.bgtasks.append(task)
                self.connectSignals(task, blocking)
                logger.debug('Starting background task')
        if self.task is None:
            try:
                self.task = self.tasks.popleft()
                self.connectSignals(self.task)
            except IndexError:
                logger.info('Completed all pending tasks')

    @pyqtSlot(QTask)
    def dequeueTask(self, task):
        """Removes completed task from task queue or background list"""
        self.disconnectSignals(task)
        if task.blocking:
            self.task = None
            self.queueTask()
        else:
            self.bgtasks.remove(task)

    @pyqtProperty(bool)
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, paused):
        self._paused = bool(paused)
        self.sigPause.emit(self._paused)

    def pauseTasks(self):
        """Toggle the pause state of the task manager"""
        self.paused = not self.paused

    def clearTasks(self):
        """Empty task queue"""
        self.tasks.clear()
        self.bgtasks.clear()
