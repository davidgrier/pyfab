# -*- coding: utf-8 -*-

'''Simple task to delay a certain number of frames'''

from .Task import Task


class Delay(Task):
    def __init__(self, **kwargs):
        # pass in delay as keyword
        super(Delay, self).__init__(**kwargs)
