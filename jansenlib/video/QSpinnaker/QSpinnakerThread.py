# -*- coding: utf-8 -*-

"""QSpinnakerThread.py: Spinnaker video camera running in a QThread"""

from PyQt5.QtCore import (QThread, pyqtSignal, pyqtSlot)
from SpinnakerCamera import SpinnakerCamera as Camera
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class QSpinnakerThread(QThread):

    """Spinnaker camera

    Continuously captures frames from a video camera,
    emitting sigNewFrame when each frame becomes available.

    NOTE: Subclassing QThread is appealing for this application
    because reading frames is blocking and I/O-bound, but not
    computationally expensive.  QThread moves the read operation
    into a separate thread via the overridden run() method
    while other methods and properties remain available in
    the calling thread.  This simplifies getting and setting
    the camera's properties.
    """

    sigNewFrame = pyqtSignal(np.ndarray)

    def __init__(self, parent=None, **kwargs):
        super(QSpinnakerThread, self).__init__(parent)

        self.camera = Camera(**kwargs)

    def run(self):
        logger.debug('Starting acquisition loop')
        self._running = True
        while self._running:
            ready, frame = self.camera.read()
            if ready:
                self.sigNewFrame.emit(frame)
        del self.camera

    def stop(self):
        self._running = False

    def size(self):
        return (self.camera.height, self.camera.width)
