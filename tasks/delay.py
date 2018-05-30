# -*- coding: utf-8 -*-

'''Simple task to delay a certain number of frames'''

from task import task

class delay(task):

    def __init__(self, **kwargs):
        super(delay, self).__init__(delay=100, **kwargs)
