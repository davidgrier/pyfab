# -*- coding: utf-8 -*-

"""
Brownian molecular dynamics simulation for moving
a set of traps to a set of targets
"""

from .TrapMove import TrapMove, Trajectory
from PyQt5.QtCore import pyqtProperty
import numpy as np
import itertools
import heapq


class TrapAssemble(TrapMove):

    def __init__(self, **kwargs):
        super(TrapAssemble, self).__init__(**kwargs)

        self._targets = None

        self._padding = 1.5  # [um]
        self._tsteps = 300
        self._zrange = (-5, 10)   # [um]

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
        '''Spacing between traps. Used for graph discretization [um]'''
        return self._padding

    @padding.setter
    def padding(self, padding):
        self._padding = padding

    @pyqtProperty(tuple)
    def zrange(self):
        '''z-range in chamber that traps can travel [um]'''
        return self._zrange

    @zrange.setter
    def zrange(self, zrange):
        self._zrange = zrange

    @pyqtProperty(int)
    def tsteps(self):
        '''Maximum number of steps to get to destination'''
        return self._tsteps

    @tsteps.setter
    def tsteps(self, steps):
        self._tsteps = steps

    #
    # Finding trajectories
    #
    def parameterize(self, traps):
        # Get tunables
        pattern = self.parent().pattern.pattern
        cgh = self.parent().cgh.device
        mpp = cgh.cameraPitch/cgh.magnification
        w, h = (self.parent().screen.source.width,
                self.parent().screen.source.height)
        zmin, zmax = (int(self.zrange[0]/mpp),
                      int(self.zrange[1]/mpp))
        tsteps = self.tsteps
        padding = int(self.padding / mpp)
        # Initialize graph w/ obstacles at all traps we ARENT moving
        # Bottom left is (0, 0)
        x = np.arange(0, w+padding, padding)
        y = np.arange(0, h+padding, padding)
        z = np.arange(zmin, zmax+padding, padding)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        G = np.full((tsteps, *xv.shape), np.inf, dtype=np.float16)
        # Find initial/final positions of traps in graph and
        # set traps nodes not in group off-limits
        group = traps.flatten()
        r_0 = {}
        r_f = {}
        for trap in pattern.flatten():
            r0 = np.array([trap.r.x(), trap.r.y(), trap.r.z()])
            i0, j0, k0 = self.locate(r0, xv, yv, zv)
            if trap not in group:
                G[:, i0, j0, k0] = -1
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
            trajectory = self.shortest_path(
                r_0[trap], r_f[trap], G, (xv, yv, zv))
            trajectories[trap] = trajectory
        # Smooth out trajectories with some reasonable step size
        print(trajectories)

    def shortest_path(self, source, target, G, rv):
        '''
        A* graph search to find shortest path between loc1
        and loc2 in G, using weights defined by rv = (xv, yv, zv)
        coordinate system.
        '''
        xv, yv, zv = rv
        trajectory = Trajectory()
        path = {}
        source = (0, *source)
        target = (G.shape[0]-1, *target)
        print("SOURCE: ", source)
        print("TARGET: ", target)
        G[source] = 0
        heap = [(0, source)]
        open = [source]
        # Case of 2 traps? What if source = target?
        while open:
            m, node = heapq.heappop(heap)
            open.remove(node)
            if node[1:] == target[1:]:
                target = node
                break
            g = G[node]
            neighbors = self.neighbors(node, target, G)
            for neighbor in neighbors:
                g_current = G[neighbor]
                g_tentative = g + self.w(node, neighbor, target)
                if g_tentative < g_current:
                    path[neighbor] = node
                    G[neighbor] = g_tentative
                    if neighbor not in open:
                        h = self.h(neighbor, target)
                        open.append(neighbor)
                        heapq.heappush(
                            heap, (g_tentative+h, neighbor))
        node = target
        t = target[0]
        trajectory.data = np.zeros((target[0]+1, 3))
        while True:
            r = np.array([xv[node[1:]], yv[node[1:]], zv[node[1:]]])
            trajectory.data[t] = r
            G[node] = -1
            print(node)
            if node == source:
                break
            node = path[node]
            t -= 1
        (tf, x, y, z) = target
        G[tf:, x, y, z] = -1
        self.reset(G)
        print(trajectory.data)
        return trajectory

    '''
    def w(self, u, v):
                Given (t, i, j, k) positions of two nodes, return
        euclidian distance between them.
        
        dr = np.array(v[1:]) - np.array(u[1:])
        return np.float16(np.sqrt(dr.dot(dr)))
    '''

    def w(self, node, neighbor, target):
        '''
        Edge length from node to neighbor, given that you're
        headed toward target
        '''
        return 0 if neighbor[1:] == target[1:] else 1

    def h(self, u, target):
        dr = np.array(target[1:]) - np.array(u[1:])
        return np.float16(np.sqrt(2*dr.dot(dr)))

    def neighbors(self, u, target, G):
        '''
        Given node (t, i, j, k), return all neighboring nodes in G.
        '''
        (t, i, j, k) = u
        (nt, nx, ny, nz) = G.shape
        if t == nt-1:
            print("DEAD END")
            return []
        elif u[1:] == target[1:]:
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
                        if G[node] != -1:
                            ds = np.absolute(
                                np.array(target)-np.array(node))
                            dt, dr = (ds[0], ds[1:])
                            if dt >= dr.sum():
                                neighbors.append(node)
            return neighbors

    def reset(self, G):
        g = G.flatten()
        idxs = np.where(g != -1)[0]
        print("Total number of Nodes: ", g.size)
        print("Number of off-limit Nodes: ", g.size-idxs.size)
        g[idxs] = np.inf
        G = g.reshape(G.shape)

    def locate(self, r, xv, yv, zv):
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
