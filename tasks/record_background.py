# -*- coding: utf-8 -*-
# MENU: Record Background

from .assemble import assemble


class record_background(assemble):
    """Move traps to nearest corners, record video, then move them back."""

    def __init__(self, **kwargs):
        super(record_background, self).__init__(**kwargs)

    def structure(self, traps):
        """Find closest edge for each trap then register
        record and return-trip tasks.
        """
        traps = traps.flatten()
        return_vertices = {}
        vertices = {}
        x_max = self.parent.screen.video.camera.width()
        y_max = self.parent.screen.video.camera.height()
        for trap in traps:
            x, y, z = trap.x, trap.y, trap.z
            d = {x: (-20, y, z), y: (x, -20, z),
                 y_max - y: (x, y_max+20, z), x_max - x: (x_max+20, y, z)}
            m = min(d.keys())
            vertices[trap] = d[m]
            return_vertices[trap] = (x, y, z)
        self.register('record', nframes=250)
        self.register('assemble', vertices=return_vertices)
        return vertices
