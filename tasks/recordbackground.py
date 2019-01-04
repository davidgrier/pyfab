# -*- coding: utf-8 -*-
# MENU: Record Background

from .guidedmove import guidedmove


class recordbackground(guidedmove):
    """Move traps to nearest corners, record video, then move them back."""

    def __init__(self, **kwargs):
        super(recordbackground, self).__init__(**kwargs)

    def calculate_targets(self, traps):
        """Find closest edge for each trap then register
        record and return-trip tasks.
        """
        traps = traps.flatten()
        return_targets = {}
        targets = {}
        x_max = self.parent.screen.video.camera.width()
        y_max = self.parent.screen.video.camera.height()
        for trap in traps:
            x, y, z = trap.x, trap.y, trap.z
            d = {x: (-20, y, z), y: (x, -20, z),
                 y_max - y: (x, y_max+20, z), x_max - x: (x_max+20, y, z)}
            m = min(d.keys())
            targets[trap] = d[m]
            return_targets[trap] = (x, y, z)
        self.register('record', nframes=250)
        self.register('guidedmove', targets=return_targets)
        return targets
