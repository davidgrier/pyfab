#!/usr/bin/env python

import cv2
from QCameraItem import QVideoItem
import os


class fabdvr(object):

    def __init__(self,
                 source=None,
                 filename='~/data/fabdvr.avi',
                 codec='HFYU', **kwds):
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
        super(fabdvr, self).__init__(**kwds)
        self._writer = None
        self.source = source
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
        if not self.hassource():
            return
        self.framenumber = 0
        self._writer = cv2.VideoWriter(self.filename,
                                       self._fourcc,
                                       self.source.device.fps,
                                       self.size(),
                                       not self.source.gray)
        self.source.sigNewFrame.connect(self.write)

    def stop(self):
        if self.isrecording():
            self.source.sigNewFrame.disconnect()
            self._writer.release()
        self.nframes = 0
        self._writer = None

    def write(self, frame):
        if self.source.transposed:
            frame = cv2.transpose(frame)
        if self.source.flipped:
            frame = cv2.flip(frame, 0)
        self._writer.write(frame)
        self.framenumber += 1
        if (self.framenumber == self._nframes):
            self.stop()

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, source):
        if isinstance(source, QVideoItem):
            self._source = source

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, filename):
        if not self.isrecording():
            self._filename = os.path.expanduser(filename)

    def hassource(self):
        return isinstance(self.source, QVideoItem)

    def isrecording(self):
        return (self._writer is not None)

    def size(self):
        if self.hassource():
            sz = self.source.device.size
            w = int(sz.width())
            h = int(sz.height())
            return (w, h)
        else:
            return None

    def framenumber(self):
        return self._framenumber


if __name__ == '__main__':
    from PyQt4 import QtGui
    from QCameraDevice import QCameraDevice
    from QVideoItem import QVideoWidget
    import sys

    app = QtGui.QApplication(sys.argv)
    device = QCameraDevice(size=(640, 480), gray=True)
    source = QVideoItem(device)
    widget = QVideoWidget(source, background='w')
    widget.show()
    dvr = fabdvr(source=source)
    dvr.record(24)
    sys.exit(app.exec_())
