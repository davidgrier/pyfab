# -*- coding: utf-8 -*-

"""
A* graph search for moving a set of traps to a set of targets


To subclass this method, just override aim() with a method which returns targets (analagous to overriding 'parameterize' in MoveTraps).
Any parameters can be passed as kwargs on __init__ and output must be a dict with keys QTrap and values tuple (x, y, z)
"""

from .Move import Move
from PyQt5.QtCore import pyqtProperty
from numba import njit
import numpy as np
import itertools
import heapq

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Assemble(Move):

    def __init__(self, targets=None, **kwargs):
        super(Assemble, self).__init__(**kwargs)
        self.delay = 4 if self.delay < 4 else self.delay

        self.targets = targets
               
        self.nframes = self.nframes or 300   #### Note: nframes=0 is not allowed, so let default be 300 
        self.smooth = True
        self._particleSpacing = 1  # [um]
        self._gridSpacing = .5     # [um]
        self._zrange = (-5, 10)    # [um]         

    def aim(self, traps):    #### Subclass to set targets
        pass

    def initialize(self, frame):
        logger.info('finding targets for {} traps...'.format(len(self.traps)))
        self.aim(self.traps)
        super(Assemble, self).initialize(frame)
        

    #
    # Setters for user interaction
    #
    @property
    def targets(self):
        '''Dictionary with QTrap keys and (x, y, z) values'''
        return self._targets

    @targets.setter
    def targets(self, targets):
        if targets is None:
            return
        if len(self.traps) == 0:
            logger.warning("Set QTraps before setting targets")
        elif type(targets) is dict:
            self._targets = dict(targets)
        else:
            print(targets)
            targets = list(targets)
            print(targets)
            logger.info("Pairing traps to targets")
            if len(self.traps) == len(targets):
#                 self.parent().screen.source.blockSignals(True)
#                 self.parent().screen.pauseSignals(True)
                self._targets = self.pair(targets)
#                 self.parent().screen.source.blockSignals(False)
#                 self.parent().screen.pauseSignals(False)
            else:
#                 logger.warning("Number of targets does not match number of traps")
                raise Exception("Number of targets does not match number of traps")

    #
    # Tunable parameters
    #
    @property
    def stepSize(self):
        return self._stepSize or self._gridSpacing

    @stepSize.setter
    def stepSize(self, stepSize):
        self._stepSize = stepSize

    @property
    def particleSpacing(self):
        '''Spacing between traps. Used for graph discretization [um]'''
        return self._particleSpacing

    @particleSpacing.setter
    def particleSpacing(self, spacing):
        self._particleSpacing = spacing

    @property
    def gridSpacing(self):
        return self._gridSpacing

    @gridSpacing.setter
    def gridSpacing(self, spacing):
        self._gridSpacing = spacing

    @property
    def zrange(self):
        '''z-range in chamber that traps can travel [um]'''
        return self._zrange

    @zrange.setter
    def zrange(self, zrange):
        self._zrange = zrange
        
    #
    # Finding trajectories
    #
    def parameterize(self, traps):

        # Get tunables
#         pattern = self.parent().pattern.pattern
#         logger.info('targets: {}'.format(self.targets))

        cgh = self.parent().cgh.device
        mpp = cgh.cameraPitch/cgh.magnification         # [microns/pixel]
        w, h = (self.parent().screen.source.width,
                self.parent().screen.source.height)     # [pixels]
        zmin, zmax = (self.zrange[0]/mpp,
                      self.zrange[1]/mpp)               # [pixels]
        tmax = self.nframes                             # [steps]
        gridSpacing = self.gridSpacing / mpp            # [pixels]
        particleSpacing = self.particleSpacing / mpp    # [pixels]
        # spacing = ceil(particleSpacing / gridSpacing)        # [steps]
        # Initialize graph w/ obstacles at all traps we ARENT moving
        logger.info('tmax is {}'.format(tmax))                  
        x = np.arange(0, w+gridSpacing, gridSpacing)
        y = np.arange(0, h+gridSpacing, gridSpacing)
        if zmax < zmin:
            gridSpacing *= -1
        z = np.arange(zmin, zmax+gridSpacing, gridSpacing)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        G = np.full((tmax, *xv.shape), np.inf, dtype=np.float32)
        # Find initial/final positions of traps in graph and
        # set traps nodes not in group off-limits
        r_0 = {}
        r_f = {}
        group = traps
#         print('our traps: {}'.format(self.traps))
#         print('total traps: {}'.format(self.parent().pattern.traps.flatten()))
#         print()
        for trap in self.parent().pattern.traps.flatten():
            r0 = np.array([trap.r.x(), trap.r.y(), trap.r.z()])
            i0, j0, k0 = self.locate(r0, xv, yv, zv)
            if trap not in group:
                path = []
                logger.info('{} not in group'.format(trap))
                for t in range(tmax):
                    path.append((t, i0, j0, k0))
                self.update(
                    G, path, particleSpacing, (xv, yv, zv))
            else:
                logger.info(self.targets)
                rf = self.targets[trap]
                i, j, k = self.locate(rf, xv, yv, zv)
                r_0[trap] = (i0, j0, k0)
                r_f[trap] = (i, j, k)
        # Make sure a target isn't blocked
        for trap in r_f.keys():
            i, j, k = r_f[trap]
            if np.isnan(G[tmax-1, i, j, k]):
                msg = 'Assemble failed. '
                msg += 'An unused trap is blocking position '
                msg += '({:.2f}, {:.2f}, {:.2f}).'
                return -1, msg.format(*self.targets[trap])
        # Sort traps by distance from targets

        def dist(trap):
            dr = np.array([xv[r_f[trap]] - xv[r_0[trap]],
                           yv[r_f[trap]] - yv[r_0[trap]],
                           zv[r_f[trap]] - zv[r_0[trap]]])
            return dr.dot(dr)
        group = sorted(group, key=dist)
#         print('our traps: {}'.format(self.traps))
#         print('total traps: {}'.format(self.parent().pattern.traps.flatten()))
#         print()
        # LOOP over all traps we are moving, finding the shortest
        # path for each with A* and then updating the graph with
        # new path as obstacle.
        trajectories = {}
        for i, trap in enumerate(group):
            logger.info('computing trajectory for trap {}'.format(i))
            r = (trap.r.x(), trap.r.y(), trap.r.z())
            source, target = (r_0[trap], r_f[trap])
            if np.isnan(G[tmax-1][target]):
                msg = 'Assemble failed. '
                msg += 'Spacing between targets is smaller than particleSpacing'
                raise Exception(msg)
#                 logger.warning(msg)
#                 return -1, msg
            trajectory, path = self.shortest_path(
                source, target, G, (xv, yv, zv))
            if trajectory is None:
                msg = 'Assemble failed (unknown error). '
                msg += 'Try adjusting tunables or increasing '
                msg += 'separation between traps.'
                raise Exception(msg)
#                 logger.warning(msg)
#                 return -1, msg
            trajectories[trap] = trajectory
            logger.info('added traj: trajectory {} is length {}'.format(i, len(trajectory)))
            if trap is not group[-1]:
                self.update(
                    G, path, particleSpacing, (xv, yv, zv))
                self.reset(G)
        # Set exact initial and final values in trajectory
        vertices = self.targets
        for trap in trajectories.keys():
            traj = trajectories[trap]
            r0 = (trap.r.x(), trap.r.y(), trap.r.z())
            traj[0] = r0
            traj[-1] = vertices[trap]
        # Set trajectories and global indices for TrapMove.move
#         print('Done: Trajectories calculated with length {}'.format([len(trajectories[trap]) for trap in self.traps]))
        self.trajectories = trajectories
#         return 0, ''

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
        trajectory = np.zeros((target[0]+1, 3))
        node = target
        t = target[0]
        path = []
        while True:
            r = np.array([xv[node[1:]], yv[node[1:]], zv[node[1:]]])
            trajectory[t] = r
            path.insert(0, node)
            if node == source:
                break
            try:
                node = previous[node]
            except KeyError:
                return None, None
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
    @njit(cache=True)
    def w(node, neighbor, target):
        '''
        Edge length from node to neighbor, given that you're
        headed toward target
        '''
        return 0 if neighbor[1:] == target[1:] else 1

    @staticmethod
    @njit(cache=True)
    def h(u, target):
        '''
        Heuristic for A* search. Choose euclidean distance
        of shortest possible time-like path to target's
        (x, y, z) position
        '''
        dr = (np.array(target[1:]) - np.array(u[1:])).astype(np.float32)
        return np.sqrt(2*dr.dot(dr))

    @staticmethod
    @njit(cache=True)
    def neighbors(u, target, G):
        '''
        Given node (t, i, j, k), return all neighboring nodes in G.
        '''
        (t, i, j, k) = u
        (nt, nx, ny, nz) = G.shape
        neighbors = []
        if t == nt-1:
            pass
        elif u[1:] == target[1:]:
            neighbors.append((t+1, *target[1:]))
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
    def update(self, G, path, particleSpacing, rv):
        '''
        Update graph by chopping out nodes in path
        and within radius of path
        '''
        xv, yv, zv = rv
        for node in path:
            t = node[0]
            ball = self.ball(node, G, particleSpacing, rv)
            G[t][ball] = np.nan

    @staticmethod
    def reset(G):
        '''Reset all active nodes in G to infinity'''
        idxs = np.where(~np.isnan(G))
        G[idxs] = np.inf

    @staticmethod
    @njit(cache=True)
    def ball(node, G, radius, rv):
        '''
        Return nodes of G inside ball centered on
        node in (x, y, z) space
        '''
        xv, yv, zv = rv
        xc, yc, zc = (xv[node[1:]], yv[node[1:]], zv[node[1:]])
        r = np.sqrt((xv-xc)**2 + (yv-yc)**2 + (zv-zc)**2)
        ball = np.where(r <= radius)
        return ball

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
        traps = self.traps
        if len(traps) == 0:
            return {}
        if len(traps) != len(targets):
            return {}
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
        print('paired: {}'.format(targets))
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
    @njit(cache=True)
    def _pair_genetic(vertices, traps):
        '''
        Genetic algorithm that finds best trap-target pairings.

        Args:
            traps: matrix of trap locations
            vertices: matrix of targets
        Returns:
            permutation of vertices's rows that best minimizes
            total distance traveled
        '''
        n = traps.shape[0]
        nrange = np.arange(0, n)
        no_generations = int(50*np.sqrt(n))
        generation_size = int(10*np.sqrt(n))
        # Buffers
        generation = np.empty((generation_size, n, 3))
        mutated = np.empty(generation.shape)
        buff = np.empty((generation_size*2, n, 3))
        score = np.empty(generation_size*2)
        # Initialize first generation
        for i in range(generation_size):
            generation[i] = np.random.permutation(vertices)
        for i in range(no_generations):
            for j in range(generation_size):
                # Mutate by swapping random indexes
                candidate = generation[j]
                mutated[j] = candidate
                k, l = (np.random.choice(nrange),
                        np.random.choice(nrange))
                ml = candidate[l].copy()
                mutated[j, l] = mutated[j, k]
                mutated[j, k] = ml
                # Gather candidates and mutated candidates
                buff[j] = mutated[j]
                buff[j+generation_size] = generation[j]
                # Assign scores of each
                score[j] = np.sum((mutated[j] - traps)**2)
                score[j+generation_size] = np.sum((candidate - traps)**2)
            # Sort candidates and mutated candidates and
            # assign new generation
            idxs = np.argsort(score)
            for j in range(generation_size):
                generation[j] = buff[idxs[j]]
        return generation[0]
