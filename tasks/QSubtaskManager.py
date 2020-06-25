# -*- coding: utf-8 -*-

from PyQt5.QtCore import (QObject, pyqtSlot, pyqtSignal, pyqtProperty)
from ..QTaskmanager import QTaskmanager
from collections import deque
import importlib

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)


class QSubtaskManager(QTaskmanager):

    """ QTaskmanager feeds frames from the video screen to any tasks from 
    the main task library. Likewise, QSubtaskManager feeds output from a 
    specific QTask to any QSubtasks in the respective sublibrary.
    
    source is a QTask (usually non-blocking) which includes its own sigNewFrame().
    
    """


    def __init__(self, source=None, sublib=None):
        super(QSubtaskManager, self).__init__(parent)
        self.source = source
        self.sublib = sublib
        self.source.sigDone.connect(self.cleanup)

    def registerTask(self, taskname, blocking=True, **kwargs):
        super(QSubtaskManager, self).registerTask(self, taskname + '.' + sublib, blocking=blocking, **kwargs)
        
    @pyqtSlot()
    def cleanup(self):
        self.clearTasks()
        self.source.sigDone.disconnect(self.cleanup)
