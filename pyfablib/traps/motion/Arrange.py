# -*- coding: utf-8 -*-

from .Assemble import Assemble
import numpy as np


class Arrange(Assemble):
    """General class for assembling some pattern. """

    def __init__(self, aim=None, param_names = [], **kwargs):
        super(Arrange, self).__init__(**kwargs)
        self.aim = aim
        self.params = []
        for name in param_names:
            qparam, ok = QInputDialog.getDouble(self.parent,
                                             'Parameters',
                                              name + ':')
            if ok:
                self.params.append(qparam)
            else:
                self.params.append(None)
    
    ## Method to determine where targets are. Override this in subclass, or it won't do anything!
    def aim(self):
        pass
        
    
    def dotask(self):
        if self.assembler.traps is not None:
            # Set tunables
            self.assembler.smooth = True
            self.assembler.stepRate = 15         # [steps/s]
            self.assembler.stepSize = .2         # [um]
            self.assembler.particleSpacing = 2   # [um]
            self.assembler.gridSpacing = .5      # [um]
            self.assembler.zrange = (5, -10)     # [um]
            self.assembler.tmax = 300            # [steps]
            
            # Set vertices and begin assembly
            self.assembler.targets = self.aim()
            self.assembler.start()
            
            
class Circle(Arrange):
    def __init__(self):
        super(Circle, self).__init__(param_names=['r'], **kwargs)
    
    def aim(self):
        vertices = []
        r = self.params[0] if self.params[0] is not None else 200    # pixels
        xc, yc = (self.cgh.xc, self.cgh.yc)
        ntraps = len(self.assembler.traps.flatten()
        for i in range(ntraps):
            theta = 2*np.pi*(idx+1) / ntraps
            vertices.append(np.array([xc + radius*np.cos(theta),
                                      yc + radius*np.sin(theta),
                                      0]))
            return vertices
            
            

class Rotate(Arrange):
    def __init__(self):
        super(Rotate, self).__init__(param_names=['theta'], **kwargs)
    
    def aim(self):
        vertices = []
        theta = self.params[0] if self.params[0] is not None else np.pi/6   
        cos = np.cos(theta)
        sin = np.sin(theta)
        
        traps = self.traps.flatten() 
        ntraps = len(traps)
        x = [traps.r.x for trap in traps]
        y = [traps.r.y for trap in traps]

        xc = sum(x)/ntraps
        yc = sum(y)/ntraps
        #### xc, yc = (self.cgh.xc, self.cgh.yc)   ### use this instead to rotate about origin
        for i in range(ntraps):
            xnew = xc + (x[i] - xc)*cos - (y[i] - yc)*sin
            ynew = yc + (x[i] - xc)*sin + (y[i] - yc)*cos
            vertices.append(np.array([xnew, ynew, 0]))
        return vertices

class Translate(Arrange):
    def __init__(self):
        super(Arrange, self).__init__(param_names=['theta'], **kwargs)
        
