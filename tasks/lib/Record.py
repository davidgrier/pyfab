# -*- coding: utf-8 -*-
# MENU: DVR/Record


from ..QTask import QTask


class Record(QTask):
    '''Task to record a video for a certain number of frames. 
    Attributes
    ------------
    fn : str or list
        The filename to save the video to, if provided. If a list is provided, the first element is formatted using the rest of the elements.
        For example, fn=['my{}example_{}.avi', 'rec', 1] would save the video 'my{}example{}.avi'.format('rec', 1), or 'myrecexample1.avi'.
    stop : bool
        set False to continue recording after the task is finished
    unblock : bool
        set True to queue this task as a blocking task and run it as a background task
        In general, tasks are either *blocking* (run sequentially in a queue) or *non-blocking* (run simultaneously with blocking tasks and each other).
         - If this task is blocking, it will record for nframes - during which time no other blocking task may run.
         - If this task is non-blocking, it does not wait in the queue at all - nframes are recorded, starting when the task is registered.
         - In general, a blocking task can be moved to the background at any point by setting blocking=False; in this case,
           the task continues to run as a non-blocking task and the next blocking task is queued.
        
        If blocking=True and unblock=True, then this task is switched to a non-blocking task right after initialize. 
        This allows recording to occur while blocking tasks are being executed, while still syncing the timing of the recording with a particular location in the queue.

    '''
    def __init__(self, fn=None, stop=True, unblock=True, **kwargs):
        # Pass in nframes keyword for length of recording
        super(Record, self).__init__(**kwargs)
        self.fn = fn
        self.stopVideo = stop
        self.unblock = unblock
        
        self.dvr = self.parent().dvr


    def initialize(self, frame):
        if self.unblock:
            self.blocking = False

        if self.fn is not None:
            if isinstance(self.fn, str):
                self.fn = [self.fn]
            if hasattr(self, 'loop'):
                self.fn.append(self.loop)
            fn = self.fn.pop(0)
            self.fn = fn.format(*self.fn)
            if isinstance(self.fn, str):
                self.dvr.filename = self.fn
        self.dvr.recordButton.animateClick()

    def complete(self):
        if self.stopVideo:
            self.dvr.stopButton.animateClick()
            