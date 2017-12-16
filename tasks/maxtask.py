from task import task
import numpy as np


class maxtask(task):

    def __init__(self, nframes=5):
        super(maxtask, self).__init__()
        self.nframes = nframes
        self.n = 0
        self.frame = None

    def process(self, frame):
        if self.frame is None:
            self.frame = frame
        else:
            self.frame = np.maximum(frame, self.frame)
        self.n += 1
        if self.n >= self.nframes:
            self.dotask()
            self.done = True

    def dotask(self):
        print('maxtask complete')
