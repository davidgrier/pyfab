# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QAbstractListModel, QModelIndex, pyqtSlot, pyqtSignal, pyqtProperty) 
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from .QTask import QTask
from collections import deque
import importlib
import json

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# class QTaskQueueManager(QAbstractListModel):
#     def __init__(self, *args, 

class QTaskmanager(QAbstractListModel):

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
        self.sources = {'camera': self.parent().screen.source.sigNewFrame, 'screen': self.parent().screen.sigNewFrame}
        self.task = None
        self.taskData = dict()
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
                                 blocking=blocking, 
                                 paused=self.paused, **kwargs)
            except ImportError as err:
                logger.error('Could not import {}: {}'.format(task, err))
                task = None
        task.name = taskname
        if task.widget is None:
            task.setDefaultWidget()
        self.queueTask(task)   
        return task
    
    def serialize(self, filename=None):
        info = [task.serialize() for task in self.tasks]
        info.extend([task.serialize() for task in self.bgtasks])
        if filename is not None:            
            with open('tasks/experiments/{}'.format(filename), 'w') as f:   #### change later so .json is included automatically                
                json.dump(info, f)
        return info    
    
    def connectSignals(self, task):
        task.sigDone.connect(lambda: self.dequeueTask(task))
        task.sigUnblocked.connect(lambda: self.moveToBackground(task))

        self.sigPause.connect(task.pause)
        self.sigStop.connect(task.stop)
        try:
            self.sources[task.source].connect(task.handleTask)
        except KeyError:
            logger.warn('Connect failed: PyFab has no source named {}'.format(task.source))
                
    def disconnectSignals(self, task):
            try:
                self.sources[task.source].disconnect(task.handleTask)
            except KeyError:
                logger.warn('Connect failed: PyFab has no source named {}'.format(task.source))
            except AttributeError:
                logger.warn('could not disconnect signals')
            # except TypeError:
            #     logger.warn('signal already disconnected')

    def setTaskData(self, task):
        for attr in list(self.taskData.keys()).copy():   ## can't pop attr's while also looping over attr's.  Instead, copy the keys (a list of strings) and loop over that
            if hasattr(task, attr):
                setattr(task, attr, self.taskData.pop(attr))
        # FIXME: Remove so that programmatically
        # queued tasks can provide data to a chain of tasks
        # if task.blocking:
        #     self.taskData.clear()

    def getTaskData(self, task):
        self.taskData.update(task.data())
        logger.debug('task returned data {}'.format(task.data()))
        logger.debug('task data is now {}'.format(self.taskData))
        
    def queueTask(self, task=None):
        """Add task to queue and activate next queued task if necessary"""
        if task:    
            self.parent().TaskPropertiesLayout.addWidget(task.widget)
            if task.blocking:
                self.tasks.append(task)             
                logger.debug('Queuing blocking task')
            else:
                self.bgtasks.append(task)
                self.connectSignals(task)
                self.setTaskData(task)
                logger.debug('Starting background task')
        if self.task is None:       
            try:               
                if self._paused:                         #### If paused while the queue is empty, queue a placeholder task so that first queued task
                    self.task = QTask(paused=True)       #### remains in queue until unpaused
                    self.task.name = 'Queue Paused'
                else:                                  
                    self.task = self.tasks.popleft()     #### Otherwise, dequeue next task
                self.connectSignals(self.task)
                self.setTaskData(self.task)
            except IndexError:
                # self.taskData.clear()
                logger.info('Completed all pending tasks')
    
        self.layoutChanged.emit()  

    @pyqtSlot(QTask)
    def dequeueTask(self, task):
        """Removes completed task from task queue or background list"""
        self.disconnectSignals(task)
        if task.blocking:
            self.getTaskData(self.task)
            self.task = None
            if 'error' in self.taskData.keys():
                logger.warning('Flushing task queue...')
                self.tasks.clear()
                logger.info('cleared')
                self.taskData.pop('error')               
            else:
                logger.info('queueing')
                self.queueTask()
        else:
            self.getTaskData(task)
            self.bgtasks.remove(task)    
        self.parent().TaskPropertiesLayout.removeWidget(task.widget)
        self.layoutChanged.emit()  
    
    @pyqtSlot(QTask)
    def moveToBackground(self, task):
        if task is self.task:
            self.task = None
        else:
            try:
                self.tasks.remove(task)
            except IndexError:
                logger.warn('Failed to move task from queue to background')
                return
        self.bgtasks.append(task)
        self.queueTask()
            
    @pyqtProperty(bool)
    def paused(self):
        return self._paused

    @paused.setter
    def paused(self, paused):
        self._paused = bool(paused)
        self.sigPause.emit(self._paused)
        self.layoutChanged.emit()

    def pauseTasks(self):
        """Toggle the pause state of the task manager"""
        self.paused = not self.paused

    def clearTasks(self):
        """Empty task queue"""
        self.tasks.clear()
        self.bgtasks.clear()
        self.layoutChanged.emit()
        
        
    def taskAt(self, index):
        tasksLen = len(self.tasks)
        bgLen = len(self.bgtasks)
        if index < bgLen:
            return self.bgtasks[index]
        elif index < bgLen + tasksLen:
            return self.tasks[tasksLen + bgLen - index - 1]
        elif index == bgLen + tasksLen:
            return self.task
        else:
            return None
    
    #### QAbstractItemModel must subclass data (tells PyQt how to display list) and rowCount (returns # of rows)
    def data(self, index, role):
#         print("where's my data?")
        if role == Qt.DisplayRole:
            task = self.taskAt(index.row())
            suffix = '*' if task is self.task else ''
            return None if task is None else '{}) {}{}'.format(str(index.row()), task.name, suffix)
        elif role == Qt.FontRole:
            task = self.taskAt(index.row())
            if task is None:
                return None
            font = QFont()
            font.setBold(not task._blocking)
            font.setItalic(task._paused)
            return font
        
    def rowCount(self, index):
        return len(self.tasks)+len(self.bgtasks)+1
    
    @pyqtSlot()   #### Toggle pause for all selected tasks
    def toggleSelected(self):
        tasks = [self.taskAt(index.row()) for index in self.parent().TaskManagerView.selectedIndexes()]
        state = all([task._paused for task in tasks])
        for task in tasks:
            task._paused = not state
        self.layoutChanged.emit()
        
    @pyqtSlot()   #### Toggle pause for current task
    def toggleCurrent(self):
        task = self.taskAt(self.parent().TaskManagerView.currentIndex().row())
        if task is None: return
        task._paused = not task._paused
        self.layoutChanged.emit()
  
    @pyqtSlot()   #### Dequeue all selected tasks
    def removeSelected(self):
        for index in self.parent().TaskManagerView.selectedIndexes():
            self.dequeueTask(self.taskAt(index.row()))
        self.layoutChanged.emit()

    @pyqtSlot()    #### Switch the task properties widget to that of the selected task
    def displayProperties(self):
        task = self.taskAt(self.parent().TaskManagerView.currentIndex().row())
        if task is None: return
        print(task.__dict__)

        self.parent().TaskPropertiesLayout.setCurrentWidget(task.widget)
            
 