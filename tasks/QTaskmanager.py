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

    def __init__(self, parent=None):
        super(QTaskmanager, self).__init__(parent)
        self.source = self.parent().screen.source
        self.task = None
        self.queue = deque()
        self.bgtasks = []
        self._paused = False

    def registerTask(self, task, blocking=True, **kwargs):
        """Places the named task into the task queue."""
        if isinstance(task, str):
            try:
                taskmodule = importlib.import_module('tasks.lib.' + task)
                taskclass = getattr(taskmodule, task)
                task = taskclass(parent=self.parent(), **kwargs)
            except ImportError as err:
                logger.error('Could not import {}: {}'.format(task, err))
                return
        if blocking:
            self.queue.append(task)
            self.activateTask()
        else:
            self.bgtasks.append(task)
            self.connectTask(task)

    def connectTask(self, task):
        """Connect task to signals"""
        task.sigDone.connect(self.deactivateTask)
        self.sigPause.connect(task.pause)
        self.source.sigNewFrame.connect(task.handleTask)
    
    def disconnectTask(self, task):
        """Disconnect task from signals"""
        try:
            self.source.sigNewFrame.disconnect(task.handleTask)
        except AttributeError:
            logger.warn('task destroyed before cleanup')
        
    def activateTask(self):
        """Add blocking task to queue"""
        if self.task is None:
            try:
                self.task = self.queue.popleft()
                self.connectTask(self.task)
            except IndexError:
                logger.info('Completed all pending tasks')
        
    @pyqtSlot()
    def deactivateTask(self, task=None):
        """Remove task from queue/list (for blocking/non-blocking tasks respectively"""
        if task is None:
            self.disconnectTask(self.task)
            self.task = None
            self.activateTask()
        else:
            self.disconnectTask(task)
            self.bgtasks.remove(task)

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
