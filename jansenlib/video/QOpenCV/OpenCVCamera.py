#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''QOpenCVCamera: OpenCV video camera'''

import cv2
import time

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OpenCVCamera(object):
    '''OpenCV camera

    Attributes
    ----------

    Methods
    -------
    read():
        Returns image as numpy.ndarray

    '''

    def __init__(self,
                 cameraID=0,
                 mirrored=False,
                 flipped=True,
                 gray=False):
        
        self.device = cv2.VideoCapture(cameraID)

        if cv2.__version__.startswith('2.'):
            self._WIDTH = cv2.cv.CV_CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.cv.CV_CAP_PROP_FRAME_HEIGHT
            self._toRGB = cv2.cv.CV_BGR2RGB
            self._toGRAY = cv2.cv.CV_BGR2GRAY
        else:
            self._WIDTH = cv2.CAP_PROP_FRAME_WIDTH
            self._HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
            self._toRGB = cv2.COLOR_BGR2RGB
            self._toGRAY = cv2.COLOR_BGR2GRAY

        # camera properties
        self.mirrored = bool(mirrored)
        self.flipped = bool(flipped)
        self.gray = bool(gray)

        # initialize camera with one frame
        while True:
            ready, image = self.read()
            if ready:
                break

    def read(self):
        ready, image = self.device.read()
        if not ready:
            time.sleep(0.01)
            return ready, None
        if image.ndim == 3:
            image = cv2.cvtColor(image, self._conversion)
        if self.flipped or self.mirrored:
            image = cv2.flip(image, self.mirrored * (1 - 2 * self.flipped))
        return ready, image

    # Camera properties
    @property
    def width(self):
        return int(self.device.get(self._WIDTH))

    @width.setter
    def width(self, width):
        self.device.set(self._WIDTH, width)
        logger.info('Setting camera width: {}'.format(width))

    @property
    def height(self):
        return int(self.device.get(self._HEIGHT))

    @height.setter
    def height(self, height):
        self.device.set(self._HEIGHT, height)
        logger.info('Setting camera height: {}'.format(height))

    @property
    def gray(self):
        gray = self._conversion == self._toGRAY
        logger.debug('Getting gray: {}'.format(gray))
        return gray

    @gray.setter
    def gray(self, gray):
        logger.debug('Setting gray: {}'.format(gray))
        self._conversion = self._toGRAY if gray else self._toRGB


if __name__ == '__main__':
    cam = OpenCVCamera()
    ready, image = cam.read()
    if ready:
        print(image.shape)
    del cam
