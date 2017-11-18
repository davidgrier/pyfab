#!/usr/bin/env python

import cv2
from QCameraItem import QCameraItem
import os


class fabdvr(object):

    def __init__(self, camera=None,
                 filename='~/data/fabdvr.avi',
                 codec='HFYU'):
        """Record digital video stream with lossless compression

        :param camera: object reference to QCameraItem
        :param filename: video file name.  Extension determines container.
        ;    Not all containers work with all codecs.
        :param codec: FOURCC descriptor for video codec
        :returns: 
        :rtype: 
        ;
        ;Note on codecs:
        ;    On macs, FFV1 appears to work with avi containers
        ;    On Ubuntu 16.04, HFYU works with avi container.
        ;        FFV1 fails silently
        ;        LAGS does not work (not found)

        """
        self._writer = None
        self.camera = camera
        self.filename = filename
        self._framenumber = 0
        self._nframes = 0
        if cv2.__version__.startswith('2'):
            self._fourcc = cv2.cv.CV_FOURCC(*codec)
        else:
            self._fourcc = cv2.VideoWriter_fourcc(*codec)

    def record(self, nframes=100):
        if (nframes > 0):
            self._nframes = nframes
            self.start()

    def start(self):
        if not self.hascamera():
            return
        self.framenumber = 0
        try:
            self._writer = cv2.VideoWriter(self.filename,
                                           self._fourcc,
                                           self.camera.device.fps,
                                           self.size(),
                                           not self.camera.device.gray)
        except:
            print('did not get video writer!')
            raise
        self.camera.sigNewFrame.connect(self.write)

    def stop(self):
        if self.isrecording():
            self.camera.sigNewFrame.disconnect()
            self._writer.release()
            print('recording stopped')
        self.nframes = 0
        self._writer = None

    def write(self, frame):
        img = cv2.transpose(frame)
        img = cv2.flip(img, 0)
        self._writer.write(img)
        self.framenumber += 1
        if (self.framenumber == self._nframes):
            self.stop()

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, camera):
        if isinstance(camera, QCameraItem):
            self._camera = camera

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not self.isrecording():
            self._filename = os.path.expanduser(filename)

    def hascamera(self):
        return isinstance(self.camera, QCameraItem)

    def isrecording(self):
        return (self._writer is not None)

    def size(self):
        if self.hascamera():
            sz = self.camera.size
            w = int(sz.width())
            h = int(sz.height())
            return (w, h)
        else:
            return None

    def framenumber(self):
        return self._framenumber


if __name__ == '__main__':
    from PyQt4 import QtGui
    from QCameraItem import QCameraDevice, QCameraWidget
    import sys

    app = QtGui.QApplication(sys.argv)
    device = QCameraDevice(size=(640, 480), gray=True)
    camera = QCameraItem(device)
    widget = QCameraWidget(camera, background='w')
    widget.show()
    dvr = fabdvr(camera=camera)
    dvr.record(24)
    sys.exit(app.exec_())
