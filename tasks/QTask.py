# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSlot, pyqtSignal, pyqtProperty)
from PyQt5.QtWidgets import QWidget, QLabel, QLineEdit, QFormLayout

from common.QSettingsWidget import QSettingsWidget 

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

    def __init__(self, delay=0, nframes=0, skip=1, paused=False, blocking=True, **kwargs):
        super(QTask, self).__init__(**kwargs)
        self._blocking = blocking
        self.delay = delay
        self.nframes = nframes
        self.skip = skip
        self._initialized = False
        self._paused = paused
        self._busy = False
        self._frame = 0
        self._data = dict()
        
        self.register = None if self.parent() is None else self.parent().tasks.registerTask 
        self.widget = None
        
    def setDefaultWidget(self):
        #### If self.widget is replaced by subclass, do we need to worry about properly deleting it? (i.e. deleteLater()) 
        self.widget = QSettingsWidget(parent=None, device=self, ui=defaultTaskUi(self), include=list(self.__dict__.keys())) 
    
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
    
    @blocking.setter                      #### Call setter to move background task to background
    def blocking(self, blocking):
        if self.blocking and not blocking:
            self._blocking = False
            self.sigUnblocked.emit()

    @pyqtSlot(np.ndarray)
    def handleTask(self, frame):
        logger.debug('Handling Task')
        logger.info('{}: {}'.format(self.name, type(frame)))
        try:
            self._handleTask(frame)
        except Exception as ex:
            self._busy = True
            logger.warning('Killing task : {}'.format(ex))
            data = self.data()
            data['error'] = ex
            self.setData(data)
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
    
    def serialize(self, filename=None):   #### Save name and configurable settings
        info = self.widget.settings
        info['name'] = self.name
        if filename is not None:
            with open('tasks/experiments/'+filename, 'w') as f:
                json.dump(info, f)
        return info    
        
class defaultTaskUi(object):
    def __init__(self, task):
        self.task = task
        
    def setupUi(self, wid):
        self.layout = QFormLayout(wid)       
        keys = list(self.task.__dict__.keys())
        keys.remove('nframes'); keys.append('nframes');  ## Move common properties to the top of the form
        keys.remove('skip'); keys.append('skip');
        keys.remove('delay'); keys.append('delay'); 
        for key in ['register', 'name', 'widget', '_blocking', '_initialized', '_frame', '_data', '_busy']:
            keys.remove(key)
        for key in ['_traps', '_trajectories']:
            if key in keys:
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

    