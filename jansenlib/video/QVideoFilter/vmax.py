# -*- coding: utf-8 -*-

"""Efficient running maximum (deflicker) filter."""

import numpy as np


class vmax(object):

    def __init__(self, order=0, shape=None):
        """Compute running maximum of a video stream

        :param order: depth of deflicker filter: 2^(order + 1) images
        :param dimensions: (width, height) of images
        :returns:
        :rtype:

        """
        self.child = None
        self.shape = shape
        self.order = order
        self.index = 0

    def filter(self, data):
        self.add(data)
        return self.get()

    def get(self, reshape=True):
        """Return current median image

        :returns: deflickered image
        :rtype: numpy.ndarray

        """
        data = self.buffer.max(axis=0).astype(np.uint8)
        if reshape:
            data = np.reshape(data, self.shape)
        return data

    def add(self, data):
        """include a new image in the maximum calculation

        :param data: image data
        :returns:
        :rtype:

        """
        if data.shape != self.shape:
            self.shape = data.shape
        if isinstance(self.child, vmax):
            self.child.add(data)
            if (self.child.index == 0):
                self.buffer[self.index, :] = self.child.get(reshape=False)
                self.index = self.index + 1
        else:
            self.buffer[self.index, :] = np.ravel(data)
            self.index = self.index + 1

        if self.index == 2:
            self.index = 0
            self.initialized = True

    def reset(self):
        self.initialized = False
        if isinstance(self.child, vmax):
            self.child.reset()

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        self._shape = shape
        if shape is not None:
            self.npts = np.product(shape)
            self.buffer = np.zeros((2, self.npts), dtype=np.uint8)
            self.index = 0
            self.initialized = False
            if isinstance(self.child, vmax):
                self.child.shape = shape

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, order):
        self._order = np.clip(order, 0, 10)
        if (self._order == 0):
            self.child = None
        else:
            if isinstance(self.child, vmax):
                self.child.order = self._order - 1
            else:
                self.child = vmax(order=self._order - 1,
                                  shape=self.shape)
        self.initialized = False
