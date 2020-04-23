# -*- coding: utf-8 -*-

from .Task import Task

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Assemble(Task):
    """
    Takes the last QTrapGroup created by user and gives it
    to Pyfab's assembler. Subclassses will set the assembler's
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
    
    #### Method to set assembler tunables and determine where targets are. Override this in subclass, or it won't do anything!
    def aim(self):
        pass
    
    #### Method to set assembler tunables (stepRate, smooth, etc), and declare any parameters needed in 'aim' (i.e. a circle's radius, etc)
    def config(self):
        pass
    
    def dotask(self):
        if self.assembler.traps is not None:
            self.config()   # Configure assembler + declare 'aim' parameters
            self.assembler.targets = self.aim()
            self.assembler.start()
            
def prompt(str):
    qparam, ok = QInputDialog.getDouble(self.parent,'Parameters', str)
        if ok:
            return qparam
        else:
            return None

        
#### Example of how to subclass:
class Circle(Assemble):
    
    def config(self):
         # Set tunables
        self.assembler.smooth = True
        self.assembler.stepRate = 15         # [steps/s]
        self.assembler.stepSize = .2         # [um]
        self.assembler.particleSpacing = 2   # [um]
        self.assembler.gridSpacing = .5      # [um]
        self.assembler.zrange = (5, -10)     # [um]
        self.assembler.tmax = 300            # [steps]
        
         # Set 'aim' parameters
        self.r = 200
         # Or, prompt the user for an input:
        self.r = prompt('radius (pixels):')
         # Or, if you want to make sure the user gets it right, add this loop...
        emphasis = '!'
        while self.r is None:
            self.r = prompt("That's not a double - try again" + emphasis)
            emphasis = emphasis + '!'
       
                
    def aim(self):
        vertices = []
        r = self.r      #### Remember - we need to instantiate parameters in config! (Or, you can technically do it in aim)
        xc, yc = (self.cgh.xc, self.cgh.yc)
        ntraps = len(self.assembler.traps.flatten()
        for i in range(ntraps):
            theta = 2*np.pi*(idx+1) / ntraps
            vertices.append(np.array([xc + radius*np.cos(theta),
                                      yc + radius*np.sin(theta),
                                      0]))
            return vertices
                     
#### And that's it! init, dotask, initialize, etc are all defined in the parent, so you only need to override aim (and semi-optionally, config)
