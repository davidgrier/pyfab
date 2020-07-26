# -*- coding: utf-8 -*-
# MENU: Motion/Encircle

from .MoveTraps import MoveTraps
import numpy as np


class Encircle(MoveTraps):
    """Make particles move in a circle around some point"""
    
    def __init__(self, **kwargs):
        super(Encircle, self).__init__(smooth=False, stepSize=4, **kwargs)
        self.nframes = self.nframes or 60
        self.center = (500, 500)
        print('center is at {}'.format(self.center))
    
    def parameterize(self, traps):
        trajs = {}
        theta = np.linspace(0, 2.2*np.pi, self.nframes)
        xc, yc = self.center
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
        
