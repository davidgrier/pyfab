import numpy as np
from task import task
from autotrap import autotrap

class moveoff(autotrap):

    def __init__(self, **kwargs):
        super(moveoff, self).__init__(nframes=4, delay=5, **kwargs)

    def doprocess(self, frame):
        pass
