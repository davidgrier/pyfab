# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QAbstractListModel, QModelIndex, pyqtSlot, pyqtSignal, pyqtProperty) 
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QFormLayout
from PyQt5.QtGui import QFont
from common.QSettingsWidget import QSettingsWidget 

from .QTask import QTask
from collections import deque
import importlib

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
        self.source = self.parent().screen.source
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
#        except TypeError:
#            logger.warn('signal already disconnected')

    def setTaskData(self, task):
        attrs = []
        for attr in list(self.taskData.keys()).copy():   ## can't pop attr's while also looping over attr's.  Instead, copy the keys (a list of strings) and loop over that
            if hasattr(task, attr):
                attrs.append(attr)
                setattr(task, attr, self.taskData.pop(attr))
        logger.debug('Set attributes {} from task data'.format(attrs))
        # FIXME: Remove so that programmatically
        # queued tasks can provide data to a chain of tasks
        if task.blocking:
            self.taskData.clear()

    def getTaskData(self, task):
        self.taskData.update(task.data())
        logger.debug('task returned data {}'.format(task.data()))
        logger.debug('task data is now {}'.format(self.taskData))
        
    def queueTask(self, task=None):
        """Add task to queue and activate next queued task if necessary"""
        if task:
            if task._widget is None: 
                task._widget = QSettingsWidget(parent=None, device=task, ui=defaultTaskUi(task), include=task.taskProperties())
            self.parent().TaskPropertiesLayout.addWidget(task._widget)
#             index = len(self.bgtasks) + len(self.tasks) if task.blocking else len(self.bgtasks)
            if task.blocking:
#                 self.beginInsertRows(QModelIndex(), index, index))
                self.tasks.append(task)             
#                 self.endInsertRows()
                logger.debug('Queuing blocking task')
            else:
#                 self.beginInsertRows(QModelIndex(), index, index)
                self.bgtasks.append(task)
#                 self.endInsertRows()
                self.connectSignals(task)
                self.setTaskData(task)
                logger.debug('Starting background task')
        if self.task is None:
            try:               
                self.task = self.tasks.popleft()
                self.connectSignals(self.task)
                self.setTaskData(self.task)
                self.task.name += '*'
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
        self.parent().TaskPropertiesLayout.removeWidget(task._widget)
        self.layoutChanged.emit()  
        
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
            return self.tasks[index - bgLen]
        elif index == bgLen + tasksLen:
            return self.task
        else:
            return None
    
    #### QAbstractItemModel must subclass data (tells PyQt how to display list) and rowCount (returns # of rows)
    def data(self, index, role):
#         print("where's my data?")
        if role == Qt.DisplayRole:
            task = self.taskAt(index.row())
            return None if task is None else '{}) {}'.format(str(index.row()), task.name)
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
    
    
    #### Double-clicking toggles pause
    @pyqtSlot()
    def toggleSelected(self):
        task = self.taskAt(self.parent().TaskManagerView.currentIndex().row())
        if task is None: return
        task._paused = not task._paused
        self.layoutChanged.emit()
    
######  Code which sets up the property display widget. (This might be able to go to another file later)  ####
    @pyqtSlot()    
    def displayProperties(self):
        task = self.taskAt(self.parent().TaskManagerView.currentIndex().row())
        if task is None: return
        print(task.__dict__)
#        print(task._widget.__dict__)
#        print(task._widget.ui.__dict__)        
        self.parent().TaskPropertiesLayout.setCurrentWidget(task._widget)
            

class defaultTaskUi(object):
    def __init__(self, task):
        self.task = task
        
    def setupUi(self, wid):
        self.layout = QFormLayout(wid)       
        keys = self.task.taskProperties()
        keys.remove('nframes'); keys.append('nframes');  ## Move common properties to the top of the form
        keys.remove('skip'); keys.append('skip');
        keys.remove('delay'); keys.append('delay'); 
        for key in ['register', 'name', '_blocking', '_initialized', '_frame', '_data', '_busy']:
            keys.remove(key)

#         if 'traps' in keys:
#             keys.remove('traps')
#             self.promptTraps()
            
        keys.reverse()    
        for key in keys:
            label = QLabel()
            label.setText(key)
            lineEdit = QLineEdit()
            lineEdit.setText(str(getattr(self.task, key)))
            lineEdit.setObjectName(key)
            setattr(self, key, lineEdit)
            self.layout.addRow(label, lineEdit)
            
#class QLinePropEdit(QLineEdit):
#    def __init__(self, task, prop, **kwargs):
#        super(QLinePropEdit, self).__init__(**kwargs)
#        self.prop = prop
#        self.setTask(task)
#        self.returnPressed.connect(self.updateReady)
#
#    @pyqtSlot()
#    def updateReady(self):
#        self.update()
#   
#    def setTask(self, task):
#        self.update = lambda: setattr(task, self.prop, eval(self.text()))
    
#### We need to read the string into the correct type. One option is to type-cast using the current value, but this can throw 
#### errors if variables are initialized to "None" or for more complicated types, like tuples
#         attr = task.getattr(self.prop)
#         if attr is not None:
#             self.update = lambda: setattr(task, self.prop, type(attr)(self.text())

#### Use of eval is convenient to allow more complex type casting (i.e. tuples, lists, arrays, etc) but will throw a nasty error
#### if the user input has wrong syntax. (Also, this type of statement seems really sketchy security-wise; probably not best
#### practice if you're building a popular app that you don't want to get hacked)            
#         self.update = lambda: setattr(task, self.prop, eval(self.text()))
        