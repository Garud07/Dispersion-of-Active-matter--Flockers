import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class particles:
    def __init__(self, N, dist, bases, T, v0, r_int, rho_p, a, rho_f, nu, noise, scale=None, loc=None):
        self.N = N
        self.dist = dist
        self.bases = bases
        self.dim = len(bases)
        self.coord_boundaries = [basis.bounds for basis in self.bases]
        self.coord_length = [x[1] - x[0] for x in self.coord_boundaries]
        self.initialise_positions(scale=scale, loc=loc)
        self.fluids_vel = np.zeros((self.N, self.dim))
        self.vel = np.zeros((self.N, self.dim))
        self.J = np.zeros((self.N, self.dim, self.dim))
        self.initialise_stress()
        self.S = np.zeros((self.N, self.dim, self.dim))

        # MPI setup
        mesh_shape = dist.mesh.shape[0]
        if size > 1:
            if mesh_shape == 2:
                self.row_comm = dist.comm_cart.Sub([0, 1])
                self.col_comm = dist.comm_cart.Sub([1, 0])
            elif mesh_shape == 1:
                self.row_comm = dist.comm_cart.Sub([0])
                self.col_comm = dist.comm_cart.Sub([1])
        else:
            self.row_comm = dist.comm_cart.Sub([])
            self.col_comm = dist.comm_cart.Sub([])

        # Simulation parameters
        self.T = T
        self.step_count = 0
        self.r_int = r_int
        self.noise = noise
        self.tau_p = 14  # Fixed relaxation time

        # Random initial directions for orientation
        self.orientation = np.random.randn(self.N, self.dim)
        self.orientation /= np.linalg.norm(self.orientation, axis=1)[:, None]

        # Random initial speeds for propulsion
        if rank == 0:
            self.v0 = np.random.uniform(low=0.5 * v0, high=1.5 * v0, size=N)
        else:
            self.v0 = np.zeros(N)
        self.v0 = comm.bcast(self.v0, root=0)

        # RANDOM INITIAL VELOCITY FOR STOKES DRAG (vi_star) - BOTH MAGNITUDE AND DIRECTION
        if rank == 0:
            # Create random magnitudes (0.1 to 0.3)
            magnitudes = np.random.uniform(0.1, 0.3, size=N)

            # Create random directions (unit vectors)
            directions = np.random.randn(N, self.dim)
            norms = np.linalg.norm(directions, axis=1)
            directions /= norms[:, None]  # Normalize to unit vectors

            # Combine magnitude and direction
            self.vi_star = magnitudes[:, None] * directions
        else:
            self.vi_star = np.zeros((self.N, self.dim))

        self.vi_star = comm.bcast(self.vi_star, root=0)

    def initialise_positions(self, scale=None, loc=None):
        if rank == 0:
            # Uniform random distribution across the domain
            r_vec = np.random.uniform(
                low=[b[0] for b in self.coord_boundaries],
                high=[b[1] for b in self.coord_boundaries],
                size=(self.N, self.dim)
            )
        else:
            r_vec = np.zeros((self.N, self.dim))

        r_vec = comm.bcast(r_vec, root=0)
        self.positions = r_vec
        self.apply_bcs()

    def interpolate(self, F, locations):
        # (Keep original implementation)
        assert (len(locations) == self.dim)
        C = F['c'].copy()
        prod_list = []
        for coord in range(self.dim):
            modes = self.dist.local_modes(self.bases[coord])
            left = self.coord_boundaries[coord][0]
            if type(self.bases[coord]).__name__ == 'Jacobi':
                L = self.coord_length[coord]
                scale = np.ones(self.bases[coord].size) / np.sqrt(np.pi / 2)
                scale[0] = 1 / np.sqrt(np.pi)
                zi = np.array([
                    np.cos(modes * np.arccos(2 * (zs - left) / L - 1)) * scale
                    for zs in locations[coord]
                ])
            elif type(self.bases[coord]).__name__ == 'RealFourier':
                L = self.coord_length[coord]
                k = 2 * np.pi / L * (modes // 2)
                is_sin = modes % 2
                is_cos = is_sin ^ 1
                periodic_locs = left + np.mod(locations[coord] - left, L)
                zi = np.array([
                    is_cos * np.cos(k * (zs - left)) - is_sin * np.sin(k * (zs - left))
                    for zs in periodic_locs
                ])
            prod_list.append(np.squeeze(zi))
        if self.dim == 3:
            D = np.einsum('ijk,lk,lj,li->li', C, prod_list[2], prod_list[1], prod_list[0])
            D = self.row_comm.allreduce(D)
            D = np.einsum('li->l', D)
            D = self.col_comm.allreduce(D)
        else:
            D = np.einsum('ij,lj,li->l', C, prod_list[1], prod_list[0])
            D = comm.allreduce(D)
        return np.real(D)

    def get_fluid_vel(self, velocities):
        assert (len(velocities) == self.dim)
        for coord in range(self.dim):
            locs = tuple(self.positions[:, i] for i in range(self.dim))
            self.fluids_vel[:, coord] = self.interpolate(velocities[coord], locs)

    def step(self, dt, velocities):
        # 1) Sample fluid velocity at particle positions
        self.get_fluid_vel(velocities)
        # 2) Stokes-drag (inertial) update for vi_star
        self.vi_star += (self.fluids_vel - self.vi_star) * (dt / self.tau_p)
        # 3) Net velocity: Stokes drag + self propulsion
        self.vel = self.vi_star + self.v0[:, None] * self.orientation
        # 4) Vicsek alignment every T steps
        if self.step_count % self.T == 0:
            self.align_with_neighbors()
        # 5) Move particles
        self.positions += dt * self.vel
        # 6) Apply BCs
        self.apply_bcs()
        # 7) Increment counter
        self.step_count += 1

    def align_with_neighbors(self):
        new_orientation = np.zeros_like(self.orientation)
        for i in range(self.N):
            # Calculate periodic distances correctly
            delta = self.positions - self.positions[i]
            for dim in range(self.dim):
                if type(self.bases[dim]).__name__ == 'RealFourier':
                    L = self.coord_length[dim]
                    delta[:, dim] = (delta[:, dim] + L / 2) % L - L / 2

            d2 = np.sum(delta ** 2, axis=1)
            mask = (d2 < self.r_int ** 2) & (d2 > 0)  # Exclude self

            dirs = self.orientation[mask]
            if dirs.size == 0:
                new_orientation[i] = self.orientation[i]
                continue

            n_avg = dirs.mean(axis=0)
            # Add noise - CRITICAL for realistic flocking
            noise_vec = self.noise * np.random.randn(self.dim)
            n_avg += noise_vec
            norm = np.linalg.norm(n_avg)
            if norm > 1e-8:  # Avoid division by zero
                n_avg /= norm
            else:
                n_avg = self.orientation[i]
            new_orientation[i] = n_avg
        self.orientation = new_orientation

    def apply_bcs(self):
        # Apply BCs on the particle positions
        for coord in range(self.dim):
            if (type(self.bases[coord]).__name__ == 'RealFourier'):
                pass
                # Periodic boundary conditions
                # self.positions[:,coord] = self.coord_boundaries[coord][0]+np.mod(self.positions[:,coord]-self.coord_boundaries[coord][0],self.coord_length[coord])
            if (type(self.bases[coord]).__name__ == 'Jacobi'):
                # Non-periodic boundary conditions
                self.positions[:, coord] = np.clip(self.positions[:, coord], self.coord_boundaries[coord][0],
                                                   self.coord_boundaries[coord][1])

    def initialise_stress(self):
        for p in range(self.N):
            self.J[p, 0, 0] = 1. / np.sqrt(2)
            self.J[p, 1, 1] = 1. / np.sqrt(2)

    def get_fluid_stress(self, velocities):
        assert (len(velocities) == self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                diff_op = self.bases[j].Differentiate(velocities[i])  # Fixed parenthesis
                positions_tuple = tuple(self.positions[:, k] for k in range(self.dim))  # Separated for clarity
                self.S[:, i, j] = self.interpolate(diff_op, positions_tuple)  # Properly closed

    def step_stress(self, dt, velocities):
        self.get_fluid_stress(velocities)
        for p in range(self.N):
            self.J[p, :, :] += dt * (self.S[p, :, :] @ self.J[p, :, :])

    def global_order_parameter(self):
        """Return global polarization (flocking order parameter)."""
        mean_dir = np.mean(self.orientation, axis=0)
        return np.linalg.norm(mean_dir)

    def plot_flocking(self, filename="flocking_snapshot.png", max_particles=200):
        """Visualize flocking state with blue arrows and red average direction"""
        if rank != 0:
            return  # Only rank 0 creates plots

        plt.figure(figsize=(10, 8))

        # Plot domain boundaries
        xmin, xmax = self.coord_boundaries[0]
        ymin, ymax = self.coord_boundaries[1]

        # Plot subset of particles (for clarity)
        n_plot = min(self.N, max_particles)
        indices = np.random.choice(self.N, n_plot, replace=False) if self.N > max_particles else np.arange(self.N)

        # Individual particles (blue arrows)
        for i in indices:
            x, y = self.positions[i]
            dx, dy = self.orientation[i]
            plt.arrow(x, y, dx * 0.2, dy * 0.2,
                      head_width=0.05, head_length=0.1,
                      fc='blue', ec='blue', alpha=0.7)

        # Average direction (thick red arrow)
        avg_dir = np.mean(self.orientation, axis=0)
        norm = np.linalg.norm(avg_dir)
        if norm > 1e-8:
            avg_dir /= norm
        center_x = np.mean(self.positions[:, 0])
        center_y = np.mean(self.positions[:, 1])
        plt.arrow(center_x, center_y, avg_dir[0] * 0.5, avg_dir[1] * 0.5,
                  head_width=0.1, head_length=0.2,
                  fc='red', ec='red', lw=3, alpha=1.0)

        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.title(f"Flocking Order: {self.global_order_parameter():.3f}")
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.grid(True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()