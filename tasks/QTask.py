# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSlot, pyqtSignal, pyqtProperty)
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QFormLayout

from common.QSettingsWidget import QSettingsWidget 

from .DefaultTaskWidget import Ui_DefaultTaskWidget

import numpy as np
import json

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
    sigUnblocked = pyqtSignal()

    def __init__(self,
                 delay=0,
                 nframes=0,
                 skip=1,
                 paused=False,
                 blocking=True,
                 **kwargs):
        super(QTask, self).__init__(**kwargs)
        self.source = 'camera'
        self._blocking = blocking
        self.delay = delay
        self.nframes = nframes
        self.skip = skip
        self._initialized = False
        self._paused = paused
        self._busy = False
        self._frame = 0
        self._data = dict()
        self.widget = None
        if self.parent() is None:
            self.register = None
        else:
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
        # logger.debug('Cleaning up')
        pass
    
    def setData(self, data):
        self._data = data or dict()

    def data(self):
        return self._data

    @pyqtProperty(bool)
    def blocking(self):
        return self._blocking
    
    @blocking.setter
    def blocking(self, blocking):
        '''Use setter to move task to background'''
        if self.blocking and not blocking:
            self._blocking = False
            self.sigUnblocked.emit()
            self.widget.ui._blocking.setEnabled(False)
            self.widget.updateUi()

    @pyqtSlot(list)
    @pyqtSlot(np.ndarray)
    def handleTask(self, frame):
        self._handleTask(frame)

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
        try:
            self.widget.updateUi()
        except AttributeError:
            pass
        
    @pyqtSlot()
    def stop(self):
        self.shutdown()
        self.sigDone.emit()
        logger.info('TASK: {} done'.format(self.__class__.__name__))
    
    def serialize(self, filename=None):
        '''Save name and configurable settings'''
        info = self.widget.settings
        info['name'] = self.name
        if filename is not None:
            with open('tasks/experiments/'+filename, 'w') as f:
                json.dump(info, f)
        return info    
    
    # Slots for setting traps/trap groups.  
    @pyqtSlot()
    def setTraps(self):
        self.traps = self.parent().pattern.prev.flatten()
        
    @pyqtSlot()
    def selectCurrentTraps(self):
        self.parent().pattern.prev = self.group
        self.group.tree()
 
    # UI setup
    def setDefaultWidget(self):
        # If self.widget is replaced by subclass,
        # do we need to worry about properly deleting it?
        # (i.e. deleteLater()) 
        self.widget = QSettingsWidget(parent=None,
                                      device=self,
                                      ui=TaskUi(self),
                                      include=list(self.__dict__.keys()))  
   
    
class TaskUi(Ui_DefaultTaskWidget):        
    def __init__(self, task):
        super(TaskUi, self).__init__()
        self.task = task
        
    def setupUi(self, wid):
        super(TaskUi, self).setupUi(wid)
        self.layout = QFormLayout(self.settingsView)       
        keys = list(self.task.__dict__.keys())

        badkeys = ['nframes', 'skip', 'delay', 'register', 'name', 'widget', '_blocking', '_initialized', '_frame', '_data', '_paused', '_busy']
        for key in badkeys:
            keys.remove(key)

        if '_trajectories' in keys:
            keys.remove('_trajectories')
        
        if '_traps' in keys:
            keys.remove('_traps')
            self.setTraps.setEnabled(True)
            self.setTraps.clicked.connect(self.task.setTraps)
            self.showSelection.setEnabled(True)
            self.showSelection.clicked.connect(self.task.selectCurrentTraps)
             
        keys.reverse()    
        for key in keys:
            label = QLabel()
            label.setText(key)
            lineEdit = QLineEdit()
            lineEdit.setText(str(getattr(self.task, key)))
            lineEdit.setObjectName(key)
            setattr(self, key, lineEdit)
            self.layout.addRow(label, lineEdit)

    
