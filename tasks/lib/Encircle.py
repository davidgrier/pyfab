# -*- coding: utf-8 -*-
# MENU: Motion/Encircle

from .Move import Move
import numpy as np


class Encircle(Move):
    """Make particles move in a circle around some point"""
    
    def __init__(self, **kwargs):
        super(Encircle, self).__init__(smooth=False, stepSize=4, **kwargs)
        (self.xc, self.yc) = (500, 500)    #### These parameters can be overwritten by taskData
        self.nframes = self.nframes or 60
    
    def parameterize(self, traps):
        print('center is at {}'.format((self.xc, self.yc)))

        trajs = {}
        theta = np.linspace(0, 2.2*np.pi, self.nframes)
        xc, yc = (self.xc, self.yc)
        for trap in traps:
            (xi, yi, z) = (trap.x, trap.y, trap.z)
            print('trap is at {}'.format((xi, yi)))
            dx = xi - xc
            dy = yi - yc
            r = np.sqrt(np.sum(dx**2 + dy**2))
            phi = np.arctan2(dy, dx)
            print('r, theta is {}, {}*2pi'.format(r, phi/(2*np.pi)))
            x = xc + r*np.cos(theta + phi)
            y = yc + r*np.sin(theta + phi)
            trajs[trap] = [(x[i], y[i], z) for i in range(self.nframes)]
        self.trajectories = trajs
        
