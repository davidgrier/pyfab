# -*- coding: utf-8 -*-
# MENU: Record Background

from .GuidedMove import GuidedMove


class RecordBackground(GuidedMove):
    """Move traps to nearest corners, record video, then move them back."""

    def __init__(self, **kwargs):
        super(RecordBackground, self).__init__(**kwargs)

    def calculate_targets(self, traps):
        """Find closest edge for each trap then register
        record and return-trip tasks.
        """
        traps = traps.flatten()
        return_targets = {}
        targets = {}
        x_max = self.parent.screen.video.camera.width()
        y_max = self.parent.screen.video.camera.height()
        s = 20
        for trap in traps:
            x, y, z = trap.x, trap.y, trap.z
            d = {x: (-s, y, z), y: (x, -s, z),
                 y_max - y: (x, y_max+s, z), x_max - x: (x_max+s, y, z)}
            m = min(d.keys())
            targets[trap] = d[m]
            return_targets[trap] = (x, y, z)
        self.register('Record', nframes=250)
        self.register('GuidedMove', targets=return_targets)
        return targets
