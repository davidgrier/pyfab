from task import task
import numpy as np
import cv2


class maxtask(task):

    def __init__(self, nframes=10, **kwargs):
        super(maxtask, self).__init__(nframes=nframes, **kwargs)
        self.frame = None

    def doprocess(self, frame):
        if self.frame is None:
            self.frame = frame
        else:
            self.frame = np.maximum(frame, self.frame)

    def dotask(self):
        fn = self.parent.config.filename(prefix='maxtask', suffix='.png')
        cv2.imwrite(fn, self.frame)
        print('maxtask complete')
