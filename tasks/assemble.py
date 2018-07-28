# -*- coding: utf-8 -*-


"""
Brownian molecular dynamics simulation for moving
a set of traps to a set of vertices
"""

from .parameterize import parameterize, Curve
import numpy as np
import itertools


class assemble(parameterize):

    def __init__(self, **kwargs):
        super(assemble, self).__init__(**kwargs)

    def initialize(self, frame):
        self.traps = self.parent.pattern.pattern
        self.vertices = self.structure(self.traps)
        self.trajectories = self.parameterize(self.traps,
                                              vertices=self.vertices)

    def structure(self, traps):
        '''
        Returns vertices of shape to assemble. Overwrite
        in subclass to assemble specific structure.

        Args:
            traps: QTrapGroup of all traps in QTrappingPattern
        Returns:
            dictionary where keys are QTraps and values are
            ndarray vertex locations
            OR
            list of ndarray vertex locations
        '''
        return None

    def parameterize(self, traps, vertices=None):
        '''
        Returns dictionary where Keys are QTraps and Values
        are Curve objects leading to each trap's respective
        vertex.

        Args:
            traps: QTrapGroup of all traps in QTrappingPattern.
        Keywords:
            vertices: Dictionary where Keys are QTraps and Values
                      are 3D ndarray position vectors.
        '''
        trajectories = None
        if vertices is not None:
            if type(vertices) is list:
                vertices = self.pair(vertices, traps)
            # Initialize trajectories, status
            trajectories = {}
            for trap in traps.flatten():
                r_i = (trap.r.x(), trap.r.y(), trap.r.z())
                trajectories[trap] = Curve(r_i)
            precision = .1
            status, done, close = self.status(trajectories, vertices,
                                              precision=precision)
            # Calculate curves
            while not (done and close):
                # Move each trap a single step
                for trap in trajectories.keys():
                    trajectory = trajectories[trap]
                    # Create a random step
                    f = lambda x: np.random.random_sample() * np.random.choice([1, -1])
                    noise = np.array(list(map(f, np.zeros(3))))
                    if status[trap] is 'far':
                        # Take a step towards final position with noise
                        dr = self.direct(trap,
                                         vertices[trap],
                                         trajectories,
                                         precision=precision) + noise
                    elif status[trap] is 'close+jiggling':
                        # If you're close enough but others aren't,
                        # jiggle around the goal
                        dr = noise
                    elif status[trap] is 'close':
                        # If everyone is close, go to goal without noise
                        dr = self.direct(trap,
                                         vertices[trap],
                                         trajectories,
                                         precision=precision)
                    elif status[trap] is 'done':
                        # Don't move if the trap has made it
                        dr = np.zeros(3, dtype=np.float_)
                    trajectory.step(dr)
                status, done, close = self.status(trajectories, vertices,
                                                  precision=precision)
        return trajectories

    def direct(self, trap, r_v, trajectories, precision=.1):
        '''
        Returns a displacement vector directing trap toward
        its vertex but away from other traps

        Args:
            trap: QTrap of interest
            r_v: ndarray 3D position vector. trap's
                 target position
            trajectories: dictionary where keys are QTraps
                          and values are Curve objects
        '''
        # Initialize variables
        padding = 7.
        max_step = 5.
        speed = 2.5
        d_v = r_v - trajectories[trap].r_f
        # Direct to vertex
        dx, dy, dz = d_v / np.linalg.norm(d_v)
        # Avoid other particles by simulating spherical force field
        for neighbor in trajectories.keys():
            if trap is not neighbor:
                d_n = trajectories[trap].r_f - trajectories[neighbor].r_f
                r = np.linalg.norm(d_n)
                theta = np.arctan2(d_n[1], d_n[0])
                phi = np.arccos(d_n[2] / r)
                p = padding ** 2
                dx += ((np.cos(theta) * np.sin(phi)) / r) * p
                dy += ((np.sin(theta) * np.sin(phi)) / r) * p
                dz += (np.cos(phi) / r) * p
        # Scale step size by dimensionless speed
        dr = np.array([dx, dy, dz])*speed
        step_size = np.linalg.norm(dr)
        # Fix step size if it gets too large
        if step_size > max_step:
            dr /= step_size
            dr *= max_step
        # When close to goal, scale down by below desired precision
        if np.linalg.norm(d_v) <= max_step:
            dr /= step_size
            dr *= precision*.9
        return dr

    def status(self, trajectories, vertices, precision=.1):
        '''
        Routine to evaluate whether trajectories have reached
        their respective vertices or not.

        Args:
            trajectories: dictionary where Keys are QTraps and Values
                          are Curve objects.
            vertices: dictionary where Keys are QTraps and Values are
                      ndarray cartesian position vectors.
        Returns:
            status: Dictionary where keys are QTraps and values are
                    'far' if trap has not reached either its 
                    vicinity or the goal itself
                    'close+jiggling' if trap has reached the vicinity of its
                    goal but others haven't
                    'close' if all traps are close but trap hasn't reached
                    its goal
                    'done' if trap has reached its goal
            done: True if all traps in status are 'done', False otherwise
            close: True if all traps in status are 'jiggling', False
                      otherwise
        '''
        status = {}
        done = False
        close = True
        for trap in trajectories.keys():
            # If not everyone has made it to range defined by
            # precision, set state to jiggling
            x_v, y_v, z_v = vertices[trap]
            x_f, y_f, z_f = trajectories[trap].r_f
            p = precision*50
            x_cond = x_v - p <= x_f <= x_v + p
            y_cond = y_v - p <= y_f <= y_v + p
            z_cond = z_v - p <= z_f <= z_v + p
            if x_cond and y_cond and z_cond:
                status[trap] = 'close+jiggling'
            else:
                status[trap] = 'far'
                close = False
        if close:
            done = True
            for trap in trajectories.keys():
                # If everyone is close enough to jiggle around
                # their goal, everyone go toward the goal
                x_v, y_v, z_v = vertices[trap]
                x_f, y_f, z_f = trajectories[trap].r_f
                p = precision
                x_cond = x_v - p <= x_f <= x_v + p
                y_cond = y_v - p <= y_f <= y_v + p
                z_cond = z_v - p <= z_f <= z_v + p
                if x_cond and y_cond and z_cond:
                    status[trap] = 'done'
                else:
                    status[trap] = 'close'
                    done = False
        return status, done, close

    def pair(self, vertex_list, traps):
        '''
        Wrapper method that determines which way to pair
        traps to vertex locaitons. Searches all possibilities
        for small trap number and uses a genetic algorithm
        for large trap number.
        
        Returns: 
            vertices: dictionary where keys are QTraps and 
                      values are their vertex pairing
        '''
        trap_list = traps.flatten()
        # Initialize matrices of vertices and trap locations
        t = []
        for idx, trap in enumerate(trap_list):
            r_t = np.array((trap.r.x(), trap.r.y(), trap.r.z()))
            t.append(r_t)
        v = np.vstack(vertex_list)
        t = np.vstack(t)
        # Determine when to switch algorithms
        limit = 8
        # Find best trap-vertex pairings
        if len(trap_list) < limit:
            best_pairing = self._pair_search(v, t, trap_limit=limit)
        else:
            best_pairing = self._pair_genetic(v, t)
        vertices = {}
        for idx, trap in enumerate(trap_list):
            vertices[trap] = best_pairing[idx]
        return vertices

    def _pair_search(self, v, t, trap_limit=8):
        '''
        Algorithm that finds best trap-vertex pairings for
        small trap limit, and tries to find the best by
        generating random permutations for large trap limit

        Args:
            t: matrix of trap locations
            v: matrix of vertices
        Keywords:
            trap_limit: the trap number where the algorithm
                        searches through random permutations
                        rather than all
        Returns:
            permutation of v's rows that best minimizes
            total distance traveled
        '''
        N = t.shape[0]
        if N < trap_limit:
            v_perms = np.asarray(list(itertools.permutations(v)))
        else:
            num_perms = 5*10**4
            f = lambda x: np.random.permutation(v)
            v_perms = np.empty((num_perms, N, 3))
            v_perms = np.asarray(list(map(f, v_perms)))
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
        Genetic algorithm that finds best trap-vertex pairings.
        
        Args:
            t: matrix of trap locations
            v: matrix of vertices
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
        d = lambda v_perm: np.sum((v_perm - t)**2)
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
    '''
    def pair_greedy(self, vertices_list, trap_list):
         
         Greedy algorithm that pairs traps to vertices.
 
         Args:
             traps: QTrapGroup of all traps in QTrappingPattern
             vertices: list of vertices
         Returns:
             vertices: dictionary where keys are QTraps and values are
                       their vertex pairing
         
         traps = traps.flatten()
         vertices = {}
         while len(trap_list) > 0 and len(vertices_list) > 0:
             # Initialize min distance and indeces where min occurs
             d_min = np.inf
             idx_t = None
             idx_v = None
             for i, trap in enumerate(trap_list):
                 r_t = np.array((trap.r.x(), trap.r.y(), trap.r.z()))
                 for j, r_v in enumerate(vertices_list):
                     d = np.linalg.norm(r_t - r_v)
                     if d < d_min:
                         d_min = d
                         idx_t = i
                         idx_v = j
            vertices[trap_list.pop(idx_t)] = vertices_list.pop(idx_v)
        return vertices
    '''
