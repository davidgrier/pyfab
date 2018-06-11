# -*- coding: utf-8 -*-

'''Simple task to delay a certain number of frames'''

from .task import task


class delay(task):

    def __init__(self, **kwargs):
        #pass in delay as keyword
        super(delay, self).__init__(**kwargs)
