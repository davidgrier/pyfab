# -*- coding: utf-8 -*-

from .Task import Task
import numpy as np

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class MoveGroup(Task):
    """
    Takes the last QTrapGroup created by user and gives it
    to Pyfab's trap mover (please, help me think of a better
    name than "mover"). Subclass and overwrite calculate_trajectories
    """

    def __init__(self, **kwargs):
        super(MoveGroup, self).__init__(**kwargs)

    def initialize(self, frame):
        '''Makes a user select a TrapGroup to do things to'''
        self.cgh = self.parent.cgh.device
        self.mover = self.parent.mover
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
        self.mover.traps = group
    
    def config(self):
        '''
        Method to set assembler tunables (stepRate, smooth, etc),
        and declare any parameters needed in 'calculate_targets' 
        (i.e. a circle's radius, etc)
        '''
        # Set Default Tunables
        
        # Decide whether to interpolate trajectories and the
        # step size for interpolation. (For this application,
        #  interpolating is not probably not useful)
        self.mover.smooth = False
        self.mover.stepSize = .2   # [um]
        # Step step rate for trap motion
        self.mover.stepRate = 15   # [steps/s]
        
    
    def calculate_trajectories(self):
        """
        Subclass this method to determine trajectories. Should return 
        a dictionary whole keys are QTraps and vals are Trajectory objects 
        (see TrapMove.py) or Nx3 numpy arrays (N 1x3 locations per trap)
        """
        
        trajectories = {}
        for trap in traps.flatten():
            r_i = (trap.r.x(), trap.r.y(), trap.r.z())
            trajectory = np.zeros(shape=(1, 3))
            trajectory[0] = r_i
            # Do something! Perhaps
            # trajectory.data = something (N, 3) shaped
            trajectories[trap] = trajectory
        return trajectories
    
    def dotask(self):
        '''
        Set tunables for motion, set calculate_trajectories
        method, and start!
        '''
        # Set mover's general method of trajectory calculation
        # (Help! I can't think of a better name than "mover"!)
        traps = self.mover.traps
        self.mover.trajectories = self.calculate_trajectories()  ## NOTE: The setter allows us to pass a numpy array instead of trajectory object
        # Start moving stuff!
        self.mover.start()

        

          

'''
# Example of how to subclass:
class SineMove(MoveGroup):

    def config(self):
        super(SineMove, self).config()
        self.mover.stepRate = 20        ## Example of how to change a tunable  
        
        # Set 'trajectory' parameters
        self.Nsteps = 20;
        self.A = 200
        self.d = 100
        # Or, prompt the user for an input:
        emphasis = '!'
        self.d, ok = QInputDialog.getDouble(self.parent, 'Parameters', 'wavelength (pixels):')
        while not ok:
            self.d, ok = QInputDialog.getDouble(self.parent, 'Parameters', 'That's not a double - try again' + emphasis)
            emphasis = emphasis + '!'

    def calculate_trajectories(self):
        vertices = []
        # Remember - we need to instantiate parameters in config! (Or, you can technically do it in aim)
        Nsteps = self.Nsteps
        A = self.A
        d = self.d
        trajs = {}
        for i, trap in enumerate(self.assembler.traps.flatten()):
            traj = np.zeros(shape=[Nsteps, 3])
            traj[:, 0] = np.linspace(0, d, num=Nsteps) + trap.r.x()
            traj[:, 1] = A*np.sin(np.linspace(0, 2*i*np.pi, num=Nsteps)) + trap.r.y()
            
            trajs[trap] = traj
        
        return trajs

# And that's it! init, dotask, initialize, etc are all defined in the parent, so you only need to override calculate_trakectories (and semi-optionally, config)
'''
