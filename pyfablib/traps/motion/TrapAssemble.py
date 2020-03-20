# -*- coding: utf-8 -*-

"""
A* graph search for moving a set of traps to a set of targets
"""

from .TrapMove import TrapMove, Trajectory
from PyQt5.QtCore import pyqtProperty
from math import ceil
from queue import Queue
import numpy as np
import itertools
import heapq

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TrapAssemble(TrapMove):

    def __init__(self, **kwargs):
        super(TrapAssemble, self).__init__(**kwargs)

        self._targets = None

        self._particleSpacing = 1  # [um]
        self._gridSpacing = .5     # [um]
        self._tmax = 300           # [steps]
        self._zrange = (-5, 10)    # [um]

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
            logger.info("Pairing traps to targets")
            self._targets = self.pair(targets)

    #
    # Tunable parameters
    #
    @pyqtProperty(float)
    def particleSpacing(self):
        '''Spacing between traps. Used for graph discretization [um]'''
        return self._particleSpacing

    @particleSpacing.setter
    def particleSpacing(self, spacing):
        self._particleSpacing = spacing

    @pyqtProperty(float)
    def gridSpacing(self):
        return self._gridSpacing

    @gridSpacing.setter
    def gridSpacing(self, spacing):
        if spacing > self.particleSpacing:
            raise ValueError(
                "Spacing between grid points must be smaller than spacing between particles.")
        self._gridSpacing = spacing

    @pyqtProperty(tuple)
    def zrange(self):
        '''z-range in chamber that traps can travel [um]'''
        return self._zrange

    @zrange.setter
    def zrange(self, zrange):
        self._zrange = zrange

    @pyqtProperty(int)
    def tmax(self):
        '''Maximum number of steps to get to destination'''
        return self._tmax

    @tmax.setter
    def tmax(self, t):
        self._tmax = t

    #
    # Finding trajectories
    #
    def parameterize(self, traps):
        # Get tunables
        pattern = self.parent().pattern.pattern
        cgh = self.parent().cgh.device
        mpp = cgh.cameraPitch/cgh.magnification              # [microns/pixel]
        w, h = (self.parent().screen.source.width,
                self.parent().screen.source.height)          # [pixels]
        zmin, zmax = (int(self.zrange[0]/mpp),
                      int(self.zrange[1]/mpp))               # [pixels]
        tmax = self.tmax                                     # [steps]
        gridSpacing = int(self.gridSpacing / mpp)            # [pixels]
        particleSpacing = int(self.particleSpacing / mpp)    # [pixels]
        spacing = ceil(particleSpacing / gridSpacing)        # [steps]
        # Initialize graph w/ obstacles at all traps we ARENT moving
        x = np.arange(0, w+gridSpacing, gridSpacing)
        y = np.arange(0, h+gridSpacing, gridSpacing)
        if zmax < zmin:
            gridSpacing *= -1
        z = np.arange(zmin, zmax+gridSpacing, gridSpacing)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        G = np.full((tmax, *xv.shape), np.inf, dtype=np.float16)
        # Find initial/final positions of traps in graph and
        # set traps nodes not in group off-limits
        group = traps.flatten()
        r_0 = {}
        r_f = {}
        for trap in pattern.flatten():
            r0 = np.array([trap.r.x(), trap.r.y(), trap.r.z()])
            i0, j0, k0 = self.locate(r0, xv, yv, zv)
            if trap not in group:
                path = []
                for t in range(tmax):
                    path.append((t, i0, j0, k0))
                self.update(G, path, spacing)
            else:
                rf = self.targets[trap]
                i, j, k = self.locate(rf, xv, yv, zv)
                r_0[trap] = (i0, j0, k0)
                r_f[trap] = (i, j, k)
        # Sort traps by distance from targets

        def dist(trap):
            dr = np.array([xv[r_f[trap]] - xv[r_0[trap]],
                           yv[r_f[trap]] - yv[r_0[trap]],
                           zv[r_f[trap]] - zv[r_0[trap]]])
            return dr.dot(dr)
        group = sorted(group, key=dist)
        # LOOP over all traps we are moving, finding the shortest
        # path for each with A* and then updating the graph with
        # new path as obstacle.
        trajectories = {}
        for trap in group:
            r = (trap.r.x(), trap.r.y(), trap.r.z())
            logger.info(
                "Calculating shortest path for position ({}, {}, {})".format(*r))
            source, target = (r_0[trap], r_f[trap])
            trajectory, path = self.shortest_path(
                source, target, G, (xv, yv, zv))
            trajectories[trap] = trajectory
            if trap is not group[-1]:
                self.update(G, path, spacing)
                self.reset(G)
        # Do any post-processing of trajectories
        self.tune(trajectories)
        # Set trajectories and global indices for TrapMove.move
        self.trajectories = trajectories
        self.t = 0
        self.tf = self.tmax

    def shortest_path(self, source, target, G, rv):
        '''
        A* graph search to find shortest path between source
        and target in densely connected G, using edges defined
        by self.neighbors, graph weights defined by self.w, and
        heuristic defined by self.h.
        '''
        # Initialize
        previous = {}
        source = (0, *source)
        target = (G.shape[0]-1, *target)
        # A star search
        G[source] = 0
        heap = [(0, source)]
        open = [source]
        while open:
            m, node = heapq.heappop(heap)
            open.remove(node)
            # Break if search reaches correct (x, y, z)
            if node[1:] == target[1:]:
                target = node
                break
            # Update priority queue and  for all neighboring nodes
            g = G[node]
            neighbors = self.neighbors(node, target, G)
            for neighbor in neighbors:
                g_current = G[neighbor]
                g_tentative = g + self.w(node, neighbor, target)
                if g_tentative < g_current:
                    previous[neighbor] = node
                    G[neighbor] = g_tentative
                    if neighbor not in open:
                        h = self.h(neighbor, target)
                        open.append(neighbor)
                        heapq.heappush(
                            heap, (g_tentative+h, neighbor))
        # Reconstruct path in (t, i, j, k) space and (x, y, z)
        xv, yv, zv = rv
        trajectory = Trajectory()
        trajectory.data = np.zeros((target[0]+1, 3))
        node = target
        t = target[0]
        path = []
        while True:
            r = np.array([xv[node[1:]], yv[node[1:]], zv[node[1:]]])
            trajectory.data[t] = r
            path.insert(0, node)
            if node == source:
                break
            node = previous[node]
            t -= 1
        # Extend path all the way up to tmax
        tmax = G.shape[0] - 1
        tf, i, j, k = target
        for t in range(tmax - tf):
            path.append((t+tf+1, i, j, k))

        return trajectory, path

    #
    # Functions that define Graph structure
    #
    @staticmethod
    def w(node, neighbor, target):
        '''
        Edge length from node to neighbor, given that you're
        headed toward target
        '''
        return 0 if neighbor[1:] == target[1:] else 1

    @staticmethod
    def h(u, target):
        '''
        Heuristic for A* search. Choose euclidean distance
        of shortest possible time-like path to target's
        (x, y, z) position
        '''
        dr = np.array(target[1:]) - np.array(u[1:])
        return np.float16(np.sqrt(2*dr.dot(dr)))

    @staticmethod
    def neighbors(u, target, G, removing=False):
        '''
        Given node (t, i, j, k), return all neighboring nodes in G.
        '''
        (t, i, j, k) = u
        (nt, nx, ny, nz) = G.shape
        if t == nt-1:
            return []
        elif u[1:] == target[1:] and not removing:
            return [(t+1, *target[1:])]
        else:
            xneighbors = [i]
            if i != nx-1:
                xneighbors.append(i+1)
            if i != 0:
                xneighbors.append(i-1)
            yneighbors = [j]
            if j != ny-1:
                yneighbors.append(j+1)
            if j != 0:
                yneighbors.append(j-1)
            zneighbors = [k]
            if k != nz-1:
                zneighbors.append(k+1)
            if k != 0:
                zneighbors.append(k-1)
            neighbors = []
            for x in xneighbors:
                for y in yneighbors:
                    for z in zneighbors:
                        node = (t+1, x, y, z)
                        if G[node] != np.nan:
                            ds = np.absolute(
                                np.array(target)-np.array(node))
                            dt, dr = (ds[0], ds[1:])
                            if dt >= dr.sum():
                                neighbors.append(node)
            return neighbors

    #
    # Updating and reseting graph for next iteration
    #
    def update(self, G, path, spacing):
        '''
        Update graph by chopping out nodes in path
        along with as many grid squares close to that
        path as in spacing
        '''
        Q = Queue()
        for node in path:
            Q.put((1, node))
        remove = []
        target = path[-1]
        while not Q.empty():
            dist, node = Q.get()
            remove.append(node)
            if dist < spacing:
                neighbors = self.neighbors(node, target, G, removing=True)
                for neighbor in neighbors:
                    Q.put((dist+1, neighbor))
        for node in remove:
            G[node] = np.nan

    @staticmethod
    def reset(G):
        '''Reset all active nodes in G to infinity'''
        g = G.flatten()
        idxs = np.where(g != np.nan)[0]
        g[idxs] = np.inf
        G = g.reshape(G.shape)

    #
    # Moving between discrete and continuous space
    #
    @staticmethod
    def locate(r, xv, yv, zv):
        '''
        Locates closest position to r in (x, y, z)
        coordinate system
        '''
        nx, ny, nz = xv.shape
        dx = xv - r[0]
        dy = yv - r[1]
        dz = zv - r[2]
        norm = np.sqrt(dx**2+dy**2+dz**2)
        idx = np.argmin(norm.flatten())
        i, j, k = np.unravel_index(idx, norm.shape)
        return (i, j, k)

    def tune(self, trajectories):
        '''
        Post process trajectories by setting 
        exact initial and final values.
        '''
        vertices = self.targets
        for trap in trajectories.keys():
            r0 = (trap.r.x(), trap.r.y(), trap.r.z())
            trajectories[trap].data[0] = np.array(r0)
            trajectories[trap].data[-1] = vertices[trap]

    #
    # Trap-target pairing
    #
    def pair(self, targets):
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

    @staticmethod
    def _pair_search(v, t):
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

    @staticmethod
    def _pair_genetic(v, t):
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
        fac = N // 3
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
