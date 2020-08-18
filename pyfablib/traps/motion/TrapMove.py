# -*- coding: utf-8 -*-

"""Framework for moving all current traps along some trajectory"""

import numpy as np
from PyQt5.QtGui import QVector3D
from PyQt5.QtCore import (pyqtSlot, pyqtProperty, QObject)
from scipy.interpolate import splprep, splev
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrapMove(QObject):

    def __init__(self, **kwargs):
        super(TrapMove, self).__init__(**kwargs)
        self._traps = None
        self._trajectories = None

        self._stepRate = 2
        self._stepSize = None
        self._smooth = True

        self._wait = None
        self._running = False
        self.t = 0
        self.tf = 0

    #
    # Essential properties and methods for user interaction
    #
    @property
    def traps(self):
        '''A QTrapGroup selected for movement'''
        return self._traps

    @traps.setter
    def traps(self, traps):
        self._traps = traps
        if traps.__class__.__name__ == 'QTrapGroup':
            traps.select(True)

    @property
    def running(self):
        '''Indicates whether automated trap movement is running'''
        return self._running

    def start(self):
        traps = self.traps
        if traps.__class__.__name__ != 'QTrapGroup':
            logger.warning("Set QTrapGroup before starting")
        else:
            # Set number of frames to wait between movements
            fps = self.parent().screen.fps
            cgh = self.parent().cgh.device
            mpp = cgh.cameraPitch/cgh.magnification  # [microns/pixel]
            stepRate = self.stepRate
            self._wait = round(fps/stepRate)
            # Wait for a few seconds to start
            seconds = 2
            self._counter = round(fps/(1./seconds))
            # Find trajectories
            logger.info("Finding trajectories")
            self.parent().screen.source.blockSignals(True)
            self.parent().screen.pauseSignals(True)
            status, msg = self.parameterize(traps)
            if status == 0 and self.smooth:
                if self.stepSize is not None:
                    stepSize = self.stepSize / mpp
                    self.interpolate(stepSize)
            self.parent().screen.source.blockSignals(False)
            self.parent().screen.pauseSignals(False)
            # Go!
            if status == 0:
                self._running = True
            else:
                logger.warning(msg)

    #
    # PyQt properties to be tuned for performance
    #
    @pyqtProperty(float)
    def stepRate(self):
        '''Number of displacements per second [steps/s]'''
        return self._stepRate

    @stepRate.setter
    def stepRate(self, stepRate):
        self._stepRate = stepRate

    @pyqtProperty(float)
    def stepSize(self):
        '''Step size in [um]'''
        return self._stepSize

    @stepSize.setter
    def stepSize(self, stepSize):
        self._stepSize = stepSize

    @pyqtProperty(bool)
    def smooth(self):
        return self._smooth

    @smooth.setter
    def smooth(self, smooth):
        self._smooth = smooth

    #
    # Properties and methods for calculating trajectories
    #
    @property
    def trajectories(self):
        '''Dictionary with QTrap keys and Trajectory values'''
        return self._trajectories

    @trajectories.setter
    def trajectories(self, trajectories):
        for trap in trajectories.keys():
            traj = trajectories[trap]
            if type(traj) is np.ndarray:
                trajectories[trap] = Trajectory(data=traj)
        self._trajectories = trajectories

    def parameterize(self, traps):
        '''
        Overrwrite in subclass to set trajectories
        '''
        self.t = 0
        self.tf = 0
        if self.trajectories is None:
            return -1, 'Trajectories not set'
        else:
            return 0, ''

    def interpolate(self, stepSize):
        '''
        Smooth out trajectories with scipy interpolation.
        '''
        trajectories = self.trajectories
        for trap in trajectories.keys():
            traj = trajectories[trap]
            target = traj.data[-1]
            L = np.sum(
                np.linalg.norm(np.diff(traj.data, axis=0), axis=1))
            npts = int(L / stepSize)
            tspace = np.linspace(0, 1, npts, endpoint=True)
            x = traj.data[:, 0]
            y = traj.data[:, 1]
            z = traj.data[:, 2]
            if x.size > 1:
                tck, u = splprep([x, y, z], s=x.size, k=1)
                xnew, ynew, znew = splev(tspace, tck)
                traj.data = np.empty((tspace.size, 3))
                traj.data[:, 0] = xnew
                traj.data[:, 1] = ynew
                traj.data[:, 2] = znew
                traj.data[-1] = target

    #
    # Properties and methods for core movement functionality
    #
    @pyqtSlot(np.ndarray)
    def move(self, frame):
        if self._running:
            done = False
            if self._counter == 0:
                if self.t < self.tf:
                    done = True
                    self.traps.blockRefresh(True)
                    for trap in self.traps.flatten():
                        trajectory = self.trajectories[trap].data
                        if self.t < trajectory.shape[0]:
                            r_t = trajectory[self.t]
                            r_t = QVector3D(*r_t)
                            trap.moveTo(r_t)
                            done = False
                    self.traps.blockRefresh(False)
                    self.parent().pattern.refresh()
                    self.t += 1
                self._counter = self._wait
            else:
                self._counter -= 1
            if (self.t == self.tf) or done:
                self._running = False


class Trajectory(object):
    '''
    Creates and manipulates a parameterized trajectory in
    cartesian coordinates
    '''

    def __init__(self, r_i=(0, 0, 0), data=None, **kwargs):
        super(Trajectory, self).__init__(**kwargs)
        if data is None:
            self.data = np.zeros(shape=(1, 3))
            self.data[0] = np.array(r_i)
        else:
            self.data = data

    @property
    def r_f(self):
        return self.data[-1]

    @property
    def r_i(self):
        return self.data[0]

    def insert(self, index, value):
        self.data = np.insert(self.data, index, value, axis=0)

    def stitch(self, trajectory):
        '''Adds another trajectory to the end of the trajectory
        '''
        self.data = np.append(self.data,
                              trajectory.data, axis=0)

    def __str__(self):
        np.set_printoptions(
            formatter={'float': lambda x: "{0:0.2f}".format(x)})
        data = [self.data.shape,
                self.r_i,
                self.r_f]
        string = "Trajectory(shape={}, r_i={}, r_f={})"
        return string.format(*data)
