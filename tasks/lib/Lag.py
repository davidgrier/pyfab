# -*- coding: utf-8 -*-
# MENU: Lag

from ..QTask import QTask
from PyQt5.QtCore import (pyqtSignal, pyqtSlot, QObject, QThread)
from PyQt5.QtWidgets import QWidget

from time import sleep

class Worker(QObject):
    
    sigDone = pyqtSignal()
    def __init__(self, lag, **kwargs):
        super(Worker, self).__init__(**kwargs)
        self.lag = lag
    
    @pyqtSlot()
    def work(self):
        sleep(self.lag)
        self.setData({'globalframe' : 47})
        self.sigDone.emit()

class Lag(QTask):
    """ A demonstration of performing a lengthy computation in a separate thread and returning the result as task data"""
    def __init__(self, lag=10., **kwargs):
        super(Lag, self).__init__(**kwargs)
        self.lag = lag
        self.nframes = 1
            
    
    def initialize(self, frame):
        self._busy = True
        self.worker = Worker(self.lag)
        self.worker.setData = self.setData
        self._thread = QThread()
        self._thread.started.connect(self.worker.work)
        self.worker.sigDone.connect(lambda: setattr(self, '_busy', False))
        self.worker.moveToThread(self._thread)
        self._thread.start()
    
    def shutdown(self):
        self._thread.quit()
        self._thread.wait()
        self._thread = None
        