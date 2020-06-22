# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSlot, pyqtSignal, pyqtProperty)
from collections import deque
import importlib

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class QTaskmanager(QObject):

    """QTaskmanager creates and manages a queue of QTask() objects
    for the pyfab/jansen system.

    Tasks are added to the queue with registerTask() and are
    performed on a first-come-first-served basis.
    Video frames are passed to the active task by handleTask().
    Once the active task is complete, it is cleaned up and replaced
    with the next task from the queue.
    """

    sigPause = pyqtSignal(bool)
    sigStop = pyqtSignal()

    def __init__(self, parent=None):
        super(QTaskmanager, self).__init__(parent)
        self.source = self.parent().screen.source
        self.task = None
        self.queue = deque()
        self._paused = False

    def registerTask(self, taskname, blocking=True, **kwargs):
        """Places the named task into the task queue."""
        if isinstance(taskname, str):
            try:
                taskmodule = importlib.import_module('tasks.lib.' + taskname)
                taskclass = getattr(taskmodule, taskname)
                task = taskclass(parent=self.parent(), **kwargs)
            except ImportError as err:
                logger.error('Could not import {}: {}'.format(task, err))
                return
        if blocking:
            self.queue.append(task)
            self.activateTask()
        else:
            self.connectSignals(task)
        return task

    def connectSignals(self, task):
        task.sigDone.connect(self.deactivateTask)
        self.sigPause.connect(task.pause)
        self.sigStop.connect(task.stop)
        self.source.sigNewFrame.connect(task.handleTask)

    def activateTask(self):
        """Take next task from queue and connect signals"""
        if self.task is None:
            try:
                self.task = self.queue.popleft()
                self.connectSignals(self.task)
            except IndexError:
                logger.info('Completed all pending tasks')

    @pyqtSlot()
    def deactivateTask(self, task=None):
        """Removes task from queue"""
        try:
            self.source.sigNewFrame.disconnect(self.task.handleTask)
        except AttributeError:
            logger.warn('task destroyed before cleanup')
        self.task = None
        self.activateTask()

    @pyqtProperty(bool)
    def paused(self):
        return self._paused

    @paused.setter(bool)
    def paused(self, paused):
        self._paused = bool(paused)
        self.sigPause.emit(self._paused)

    def pauseTasks(self):
        """Toggle the pause state of the task manager"""
        self.paused = not self.paused

    def clearTasks(self):
        """Empty task queue"""
        self.queue.clear()
