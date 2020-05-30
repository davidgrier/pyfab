# -*- coding: utf-8 -*-

from .Task import Task
from PyQt5.QtWidgets import QInputDialog

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Assemble(Task):
    """
    Takes the last QTrapGroup created by user and gives it
    to Pyfab's assembler. Subclasses will set the assembler's
    vertices. 
    """

    def __init__(self, **kwargs):
        super(Assemble, self).__init__(**kwargs)

    def initialize(self, frame):
        self.cgh = self.parent.cgh.device
        self.assembler = self.parent.assembler
        # Set traps from last QTrapGroup created
        pattern = self.parent.pattern.pattern
        group = None
        for child in reversed(pattern.children()):
            if isinstance(child, type(pattern)):
                group = child
                break
        if group is None:
            logger.warning(
                "No traps selected. Please create a QTrapGroup.")
        self.assembler.traps = group

     def config(self):
        '''
        Subclass to adjust assembler tunables (stepRate, smooth, etc),
        and/or declare any parameters needed in 'calculate_targets' 
        (i.e. a circle's radius, etc)
        '''
        
        # Set default tunables
        self.assembler.smooth = True
        self.assembler.stepRate = 15         # [steps/s]
        self.assembler.stepSize = .2         # [um]
        self.assembler.particleSpacing = 2   # [um]
        self.assembler.gridSpacing = .5      # [um]
        self.assembler.zrange = (5, -10)     # [um]
        self.assembler.tmax = 300            # [steps]
        
    def calculate_targets(self):
        '''
        Subclass this method to determine where targets are. Should return 
        a list of vertices (one 1x3 location per trap)
        '''
        pass

    
    def dotask(self):
        if self.assembler.traps is not None:
            self.config()   # Configure assembler + declare 'target' parameters
            self.assembler.targets = self.calculate_targets()
            self.assembler.start()

'''
# Example of how to subclass:
class Circle(Assemble):

    def config(self):
        super(Circle, self).config()     ## Load defaults
        self.assembler.tmax = 250        ## Example of how to set/change an assembler tunable  

        # Set 'target' parameters
        self.r = 200
        # Or, prompt the user for an input:
        emphasis = '!'
        self.r, ok = QInputDialog.getDouble(self.parent, 'Parameters', 'radius (pixels):')
        while not ok:
            self.r, ok = QInputDialog.getDouble(self.parent, 'Parameters', 'That's not a double - try again' + emphasis)
            emphasis = emphasis + '!'

    def calculate_targets(self):
        vertices = []
        # Remember - we need to instantiate parameters! (Preferably in config, but you can technically do it here, too)
        r = self.r
        xc, yc = (self.cgh.xc, self.cgh.yc)
        ntraps = len(self.assembler.traps.flatten())
        for i in range(ntraps):
            theta = 2*np.pi*(idx+1) / ntraps
            vertices.append(np.array([xc + radius*np.cos(theta),
                                      yc + radius*np.sin(theta),
                                      0]))
            return vertices

# And that's it! init, dotask, initialize, etc are all defined in the parent, so you only need to override calculate_targets (and semi-optionally, config)
'''
