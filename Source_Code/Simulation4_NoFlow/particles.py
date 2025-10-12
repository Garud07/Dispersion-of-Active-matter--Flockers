import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class empty:
    def __init__(self):
        return

class particles:
    def __init__(self,N,dist,bases,T,v0,r_int,rho_p,a,rho_f,nu,noise,scale=None,loc=None,):
        self.N =           N
        self.dist =        dist
        self.bases =       bases
        self.dim = len(bases)
        self.coord_boundaries = [basis.bounds for basis in self.bases]
        self.coord_length = [x[1] - x[0] for x in self.coord_boundaries]
        self.initialise_positions(scale=scale, loc=loc)
        self.fluids_vel = np.zeros((self.N, self.dim))
        self.vel = np.zeros((self.N, self.dim))
        self.J = np.zeros((self.N, self.dim, self.dim))
        self.initialise_stress()
        self.S = np.zeros((self.N, self.dim, self.dim))

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

        num_fourier = 0
        first_fourier = None
        for coord in range(self.dim):
            if type(self.bases[coord]).__name__ == 'RealFourier':
                num_fourier += 1
                if first_fourier is None:
                    first_fourier = coord
        self.num_fourier_directions = num_fourier
        self.first_fourier_direction = first_fourier

        self.T = T
        self.step_count = 0
        self.v0 = v0
        self.r_int = r_int
        self.tau_p = (2 * rho_p * a**2) / (9 * rho_f * nu)

        # Initialize orientations as random unit vectors
        self.orientation = np.random.randn(self.N, self.dim)
        self.orientation /= np.linalg.norm(self.orientation, axis=1)[:, None]

        # Initialize vi_star for Stokes drag (instead of mixing into .vel directly)
        self.vi_star = np.zeros((self.N, self.dim))

    def interpolate(self, F, locations):
        assert(len(locations) == self.dim)
        C = F['c'].copy()
        prod_list = []
        for coord in range(self.dim):
            modes = self.dist.local_modes(self.bases[coord])
            left = self.coord_boundaries[coord][0]
            if type(self.bases[coord]).__name__ == 'Jacobi':
                L = self.coord_length[coord]
                scale = np.ones(self.bases[coord].size) / np.sqrt(np.pi/2)
                scale[0] = 1/np.sqrt(np.pi)
                zi = np.array([
                    np.cos(modes * np.arccos(2*(zs-left)/L - 1)) * scale
                    for zs in locations[coord]
                ])
            elif type(self.bases[coord]).__name__ == 'RealFourier':
                L = self.coord_length[coord]
                k = 2*np.pi/L * (modes // 2)
                is_sin = modes % 2
                is_cos = is_sin ^ 1
                periodic_locs = left + np.mod(locations[coord] - left, L)
                zi = np.array([
                    is_cos * np.cos(k*(zs-left)) - is_sin * np.sin(k*(zs-left))
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

    def initialise_positions(self, scale=None, loc=None):
        if rank == 0:
            r_vec = np.random.random((self.dim, self.N))
        else:
            r_vec = np.zeros((self.N,))
        r_vec = comm.bcast(r_vec, root=0)
        self.positions = np.array([
            self.coord_boundaries[i][0] + self.coord_length[i]*r_vec[i]
            for i in range(self.dim)
        ]).T
        if scale is not None:
            if rank == 0:
                r_vec = np.array([
                    np.random.normal(loc=loc[i], scale=scale[i], size=(self.N,))
                    for i in range(self.dim)
                ])
                for i in range(self.dim):
                    r_vec[i,:] = self.coord_boundaries[i][0] + np.mod(
                        r_vec[i,:]-self.coord_boundaries[i][0],
                        self.coord_length[i]
                    )
            else:
                r_vec = np.zeros((self.N,))
            r_vec = comm.bcast(r_vec, root=0)
            self.positions = r_vec.T
            self.apply_bcs()

    def get_fluid_vel(self, velocities):
        assert(len(velocities) == self.dim)
        for coord in range(self.dim):
            locs = tuple(self.positions[:, i] for i in range(self.dim))
            self.fluids_vel[:,coord] = self.interpolate(velocities[coord], locs)

    def step(self, dt, velocities):
        # 1) Sample fluid velocity at particle positions
        self.get_fluid_vel(velocities)
        # 2) Stokes-drag (inertial) update for vi_star
        self.vi_star += (self.fluids_vel - self.vi_star) * (dt / self.tau_p)
        # 3) Net velocity: Stokes drag + self propulsion
        self.vel = self.vi_star + self.v0 * self.orientation
        # 4) Vicsek alignment every T steps (update orientation, not velocity)
        if self.step_count % self.T == 0:
            self.align_with_neighbors()
        # 5) Move particles
        self.positions += dt * self.vel
        # 6) Apply BCs
        self.apply_bcs()
        # 7) Increment counter
        self.step_count += 1

    def align_with_neighbors(self):
        # Vicsek alignment: average directions of neighbors within r_int, update orientation
        new_orientation = np.zeros_like(self.orientation)
        for i in range(self.N):
            d2 = np.sum((self.positions - self.positions[i])**2, axis=1)
            mask = (d2 < self.r_int**2)
            mask[i] = False
            dirs = self.orientation[mask]
            if dirs.size == 0:
                new_orientation[i] = self.orientation[i]
                continue
            n_avg = dirs.mean(axis=0)
            n_avg /= np.linalg.norm(n_avg)
            new_orientation[i] = n_avg
        self.orientation = new_orientation

    def apply_bcs(self):
        for coord in range(self.dim):
            if(type(self.bases[coord]).__name__=='RealFourier'):
                pass
            if(type(self.bases[coord]).__name__=='Jacobi'):
                self.positions[:,coord] = np.clip(self.positions[:,coord], self.coord_boundaries[coord][0], self.coord_boundaries[coord][1])

    def initialise_stress(self):
        for p in range(self.N):
            self.J[p,0,0] = 1./np.sqrt(2)
            self.J[p,1,1] = 1./np.sqrt(2)

    def get_fluid_stress(self, velocities):
        assert(len(velocities) == self.dim)
        for i in range(self.dim):
            for j in range(self.dim):
                diff_op = self.bases[j].Differentiate(velocities[i])
                self.S[:,i,j] = self.interpolate(diff_op, tuple(self.positions[:,k] for k in range(self.dim)))

    def step_stress(self, dt, velocities):
        self.get_fluid_stress(velocities)
        for p in range(self.N):
            self.J[p,:,:] += dt * (self.S[p,:,:] @ self.J[p,:,:])

    def global_order_parameter(self):
        """Return global polarization (flocking order parameter)."""
        mean_dir = np.mean(self.orientation, axis=0)
        return np.linalg.norm(mean_dir)
