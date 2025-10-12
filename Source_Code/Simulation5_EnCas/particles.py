"""Adapted from original code by Calum S. Skene and Steven M. Tobias (2023)"""

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class particles:
    def __init__(self, N, dist, bases, T, v0, r_int, rho_p, a, rho_f, nu, noise=0.1, scale=None, loc=None):
        self.N = N
        self.dist = dist
        self.bases = bases
        self.dim = len(bases)
        self.T = T  # alignment interval
        self.v0 = v0  # swim speed
        self.r_int = r_int  # interaction radius
        self.rho_p = rho_p  # particle density
        self.a = a  # particle radius
        self.rho_f = rho_f  # fluid density
        self.nu = nu  # kinematic viscosity
        self.noise = noise  # alignment noise
        self.coord_boundaries = [basis.bounds for basis in self.bases]
        self.coord_length = [x[1] - x[0] for x in self.coord_boundaries]

# Changes are made here
        # Initialize positions and orientations
        self.initialise_positions(scale=scale, loc=loc)
        self.orientations = np.random.rand(N, self.dim) - 0.5
        self.orientations = self.orientations / np.linalg.norm(self.orientations, axis=1)[:, np.newaxis]
        # Fluid interaction variables
        self.fluids_vel = np.zeros((N, self.dim))
        self.velocities = np.zeros((N, self.dim))
        # For spherical particles (α=1)
        self.alpha = 1.0
        self.tau_p = 1.0  # Particle relaxation time
# Till here

    def get_neighbors(self, i):
        """Find particles within interaction radius of particle i"""
        distances = np.linalg.norm(self.positions - self.positions[i], axis=1)
        return np.where(distances < self.r_int)[0]

    def vicsek_align(self):
        #Vicsek alignment with noise
        new_orientations = np.zeros_like(self.orientations)

        for i in range(self.N):
            neighbors = self.get_neighbors(i)
            if len(neighbors) > 0:
                # Average neighbor orientation
                avg_dir = np.mean(self.orientations[neighbors], axis=0)
                avg_dir /= (np.linalg.norm(avg_dir) + 1e-10)  # Normalize

                # Add noise
                theta = np.arctan2(avg_dir[1], avg_dir[0])
                theta += self.noise * (np.random.rand() - 0.5)

                if self.dim == 2:
                    new_orientations[i] = [np.cos(theta), np.sin(theta)]
                else:  # 3D
                    # Add spherical noise (simplified)
                    new_orientations[i] = avg_dir + self.noise * (np.random.rand(3) - 0.5)
                    new_orientations[i] /= np.linalg.norm(new_orientations[i])
            else:
                new_orientations[i] = self.orientations[i]

        self.orientations = new_orientations

    def get_effective_velocity(self):
        """Calculate net velocity: Vi* (Stokes drag) + V0*orientation"""
        for i in range(self.N):
            # Stokes drag term (Vi*) - simplified for α=1 (spherical)
            drag_vel = - (self.velocities[i] - self.fluids_vel[i]) / self.tau_p

            # Active swimming term (V0*orientation)
            active_vel = self.v0 * self.orientations[i]

            # Net velocity
            self.velocities[i] = drag_vel + active_vel

    def step(self, dt, velocities):
        """Modified step function with Vicsek alignment"""
        # Get fluid velocity at particle positions
        self.get_fluid_vel(velocities)

        # Vicsek alignment
        self.vicsek_align()

        # Calculate effective velocity
        self.get_effective_velocity()

        # Update positions
        self.positions += dt * self.velocities
        self.apply_bcs()

    def calculate_order_parameter(self):
        """Global flocking order parameter Φ_v"""
        if self.N == 0:
            return 0.0
        v_hat = self.velocities / (np.linalg.norm(self.velocities, axis=1)[:, np.newaxis] + 1e-10)
        return np.linalg.norm(np.mean(v_hat, axis=0))

    def initialise_positions(self, scale=None, loc=None):
        # Initialise using random distributed globally
        if (rank == 0):
            r_vec = np.random.random((self.dim, self.N))
        else:
            r_vec = np.zeros((self.N,))
        r_vec = comm.bcast(r_vec, root=0)

        self.positions = np.array(
            [self.coord_boundaries[i][0] + self.coord_length[i] * r_vec[i] for i in range(self.dim)]).T

        if scale != None:
            if (rank == 0):
                r_vec = np.array(
                    [np.random.normal(loc=loc[i], scale=scale[i], size=(self.N,)) for i in range(self.dim)])
                for i in range(self.dim):
                    r_vec[i, :] = self.coord_boundaries[i][0] + np.mod(r_vec[i, :] - self.coord_boundaries[i][0],
                                                                       self.coord_length[i])
            else:
                r_vec = np.zeros((self.N,))
            r_vec = comm.bcast(r_vec, root=0)
            self.positions = r_vec.T
            self.apply_bcs()

    def get_fluid_vel(self, velocities):
        assert (len(velocities) == self.dim)
        for coord in range(self.dim):
            if (self.dim == 3):
                self.fluids_vel[:, coord] = self.interpolate(velocities[coord],
                                                             (self.positions[:, 0], self.positions[:, 1],
                                                              self.positions[:, 2]))
            elif (self.dim == 2):
                self.fluids_vel[:, coord] = self.interpolate(velocities[coord],
                                                             (self.positions[:, 0], self.positions[:, 1]))

    def apply_bcs(self):
        # Apply BCs on the particle positions
        for coord in range(self.dim):
            if (type(self.bases[coord]).__name__ == 'RealFourier'):
                pass  # Periodic BCs handled in initialise_positions
            if (type(self.bases[coord]).__name__ == 'Jacobi'):
                self.positions[:, coord] = np.clip(self.positions[:, coord],
                                                   self.coord_boundaries[coord][0], self.coord_boundaries[coord][1])

    def interpolate(self, F, locations):
        assert (len(locations) == self.dim)
        C = F['c'].copy()
        prod_list = []
        for coord in range(self.dim):
            modes = self.dist.local_modes(self.bases[coord])
            left = self.coord_boundaries[coord][0]
            if (type(self.bases[coord]).__name__ == 'Jacobi'):
                L = self.coord_length[coord]
                scale = np.ones(self.bases[coord].size) / np.sqrt(np.pi / 2)
                scale[0] = 1 / np.sqrt(np.pi)
                zi = np.array([np.cos(modes * np.arccos(2 * (zs - left) / L - 1)) * scale for zs in locations[coord]])
            elif (type(self.bases[coord]).__name__ == 'RealFourier'):
                L = self.coord_length[coord]
                k = 2 * np.pi / L * (modes // 2)
                is_sin = modes % 2
                is_cos = is_sin ^ 1
                periodic_locs = self.coord_boundaries[coord][0] + np.mod(
                    locations[coord] - self.coord_boundaries[coord][0], self.coord_length[coord])
                zi = np.array(
                    [is_cos * np.cos(k * (zs - left)) - is_sin * np.sin(k * (zs - left)) for zs in periodic_locs])
            prod_list.append(np.squeeze(zi))

        if (self.dim == 3):
            D = np.einsum('ijk,lk,lj,li->li', C, prod_list[2], prod_list[1], prod_list[0], optimize=True)
            D = self.row_comm.allreduce(D)
            D = np.einsum('li->l', D, optimize=True)
            D = self.col_comm.allreduce(D)
        elif (self.dim == 2):
            D = np.einsum('ij,lj,li->l', C, prod_list[1], prod_list[0], optimize=True)
            D = comm.allreduce(D)
        return np.real(D)

    # Stress-related methods (keep if needed)
    def initialise_stress(self):
        for particle in range(self.N):
            self.J[particle, 0, 0] = 1. / np.sqrt(2)
            self.J[particle, 1, 1] = 1. / np.sqrt(2)

    def get_fluid_stress(self, velocities):
        assert (len(velocities) == self.dim)
        for coordi in range(self.dim):
            for coordj in range(self.dim):
                diff_op = self.bases[coordj].Differentiate(velocities[coordi])
                self.S[:, coordi, coordj] = self.interpolate(diff_op,
                                                             self.positions[:, 0], self.positions[:, 1],
                                                             self.positions[:, 2])

    def step_stress(self, dt, velocities):
        self.get_fluid_stress(velocities)
        for particle in range(self.N):
            self.J[particle, :, :] += dt * self.S[particle, :, :] @ self.J[particle, :, :]

