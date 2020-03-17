# -*- coding: utf-8 -*-

"""
Brownian molecular dynamics simulation for moving
a set of traps to a set of targets
"""

from .TrapMove import TrapMove, Trajectory
from PyQt5.QtCore import pyqtProperty
import numpy as np
import itertools


class TrapAssemble(TrapMove):

    def __init__(self, **kwargs):
        super(TrapAssemble, self).__init__(**kwargs)

        self._targets = None

        self._padding = 1.  # [um]

    #
    # Setters for user interaction
    #
    @property
    def targets(self):
        '''Dictionary with QTrap keys and (x, y, z) values'''
        return self._targets

    @targets.setter
    def targets(self, targets):
        if type(targets) is dict:
            self._targets = dict(targets)
        else:
            targets = list(targets)
            self._targets = self._pair(targets)

    #
    # Tunable parameters
    #
    @pyqtProperty(float)
    def padding(self):
        '''Spacing between traps [um]'''
        return self._padding

    @padding.setter
    def padding(self, padding):
        self._padding = padding

    #
    # Finding trajectories
    #
    def parameterize(self, traps):
        self._t = 0
        self._tf = 0
        trajectories = {}
        for trap in traps.flatten():
            r_i = (trap.r.x(), trap.r.y(), trap.r.z())
            trajectories[trap] = Trajectory(r_i)
        self._trajectories = trajectories

    #
    # Trap-target pairing
    #
    def _pair(self, targets):
        '''
        Wrapper method that determines which way to pair
        traps to vertex locaitons. Searches all possibilities
        for small trap number and uses a genetic algorithm
        for large trap number.

        Returns: 
            targets: dictionary where keys are QTraps and 
                      values are their vertex pairing
        '''
        targets = list(targets)
        traps = self.traps.flatten()
        if self.traps.count() == 0:
            return {}
        if len(traps) != len(targets):
            raise ValueError(
                "Number of traps must be same as number of targets")
        # Initialize matrices of targets and trap locations
        t = []
        for idx, trap in enumerate(traps):
            r_t = np.array((trap.r.x(), trap.r.y(), trap.r.z()))
            t.append(r_t)
        v = np.vstack(targets)
        t = np.vstack(t)
        # Determine when to switch algorithms
        limit = 8
        # Find best trap-target pairings
        if len(traps) < limit:
            pairing = self._pair_search(v, t)
        else:
            pairing = self._pair_genetic(v, t)
        targets = {}
        for idx, trap in enumerate(traps):
            targets[trap] = pairing[idx]
        return targets

    def _pair_search(self, v, t):
        '''
        Algorithm that finds best trap-target pairings for
        small trap limit

        Args:
            t: matrix of trap locations
            v: matrix of targets
        Returns:
            permutation of v's rows that best minimizes
            total distance traveled
        '''
        N = t.shape[0]
        v_perms = np.asarray(list(itertools.permutations(v)))
        d_min = np.inf
        i_min = None
        for i, v_perm in enumerate(v_perms):
            d = np.sum((v_perm - t)**2)
            if d < d_min:
                d_min = d
                i_min = i
        return v_perms[i_min]

    def _pair_genetic(self, v, t):
        '''
        Genetic algorithm that finds best trap-target pairings.

        Args:
            t: matrix of trap locations
            v: matrix of targets
        Returns:
            permutation of v's rows that best minimizes
            total distance traveled
        '''
        N = t.shape[0]
        fac = N // 5 if N >= 5 else 1
        # Init number of generations, size of generations, first generation
        total_gens = 120*fac
        gen_size = 40*fac
        gen = np.asarray(list(map(lambda x: np.random.permutation(v),
                                  np.empty((gen_size, N, 3)))))
        mutated_gen = np.empty((gen_size*2, N, 3))
        # Define fitness metric

        def d(v_perm): return np.sum((v_perm - t)**2)
        for gen_idx in range(total_gens):
            mutations = np.empty(gen.shape)
            for idx, mutation in enumerate(mutations):
                # Mutate by swapping random indexes
                mutations[idx] = gen[idx]
                i, j = (np.random.choice(range(N)),
                        np.random.choice(range(N)))
                mutations[idx][[i, j]] = mutations[idx][[j, i]]
                # Mutate by reflection
                np.flipud(mutations[idx])
            # Fill mutated_gen with current gen and all mutations
            mutated_gen[:gen_size] = gen
            mutated_gen[gen_size:] = mutations
            # Cut out worst performing permutations
            gen = np.asarray(sorted(mutated_gen,
                                    key=d))
            gen = gen[:gen_size]
        return gen[0]
