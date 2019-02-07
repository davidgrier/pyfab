# -*- coding: utf-8 -*-

"""QSpinnakerThread.py: Spinnaker video camera running in a QThread"""

from pyqtgraph.Qt import QtCore
from SpinnakerCamera import SpinnakerCamera as Camera
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class QSpinnakerThread(QtCore.QThread):

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

    NOTE: This implementation only moves the camera's read()
    method into a separate thread, not the entire camera.
    FIXME: Confirm that this is acceptable practice.
    """

    sigNewFrame = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, parent=None, **kwargs):
        super(QSpinnakerThread, self).__init__(parent)

        self.camera = Camera(**kwargs)
        self.read = self.camera.read
        ready, self.frame = self.read()

    def run(self):
        self.running = True
        while self.running:
            ready, frame = self.read()
            if ready:
                self.sigNewFrame.emit(frame)
        del self.camera

    def stop(self):
        self.running = False

    def getProperty(self, name):
        return self.camera.getProperty(name)

    @QtCore.pySlot(object, object)
    def setProperty(self, name, value):
        self.camera.setProperty(name, value)
