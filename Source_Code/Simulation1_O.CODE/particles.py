"""Adapted from original code by Calum S. Skene and Steven M. Tobias (2023)"""

import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class empty:
    def __init__(self):
        return

class particles:
    def __init__(self,N,dist, bases, scale=None, loc=None):
        # N - number of particles
        # dist - distribution
        # bases - space Fourier/Chebyshev
        # domain - domain object from the simulation
        self.N               = N
        self.dist            = dist
        self.bases           = bases
        self.dim             = len(bases)
        self.coord_boundaries = [basis.bounds for basis in self.bases]
        self.coord_length     = [x[1] - x[0] for x in self.coord_boundaries]
        # Initialise
        self.initialise_positions(scale=scale, loc=loc)
        self.fluids_vel = np.zeros((self.N,self.dim))

        self.J = np.zeros((self.N,self.dim,self.dim))
        self.initialise_stress()
        self.S = np.zeros((self.N,self.dim,self.dim))

        mesh_shape = dist.mesh.shape[0]
        
        # Be fancy and make new comms
        if(size>1):
            if(mesh_shape==2):
                self.row_comm = dist.comm_cart.Sub([0,1])
                self.col_comm = dist.comm_cart.Sub([1,0])
            elif(mesh_shape==1):
                self.row_comm = dist.comm_cart.Sub([0])
                self.col_comm = dist.comm_cart.Sub([1])
        else:
            self.row_comm = dist.comm_cart.Sub([])
            self.col_comm = dist.comm_cart.Sub([])

        # Work out if there are Fourier directions, and which is first
        num_fourier_directions = 0
        first_fourier_direction = 4
        for coord in range(self.dim):
            if(type(self.bases[coord]).__name__=='RealFourier'):
                num_fourier_directions += 1
                if(first_fourier_direction==4):
                    first_fourier_direction = coord
        self.num_fourier_directions = num_fourier_directions
        self.first_fourier_direction = first_fourier_direction

    def interpolate(self,F, locations):

        assert(len(locations)==self.dim)
        # Based on code in the Dedalus users group

        # Coefficients
        C = F['c'].copy()

        prod_list = []
        for coord in range(self.dim):
            modes = self.dist.local_modes(self.bases[coord])
            left = self.coord_boundaries[coord][0]
            if(type(self.bases[coord]).__name__=='Jacobi'):
                L = self.coord_length[coord]
                scale = np.ones(self.bases[coord].size)/np.sqrt(np.pi/2)
                scale[0] = 1/np.sqrt(np.pi)
                zi = np.array([np.cos(modes*np.arccos(2*(zs-left)/L-1))*scale for zs in locations[coord]])
                
            elif(type(self.bases[coord]).__name__=='RealFourier'):
                L = self.coord_length[coord]
                k = 2*np.pi/L*(modes // 2)
                is_sin = modes%2
                is_cos = is_sin^1
                periodic_locs = self.coord_boundaries[coord][0]+np.mod(locations[coord]-self.coord_boundaries[coord][0],self.coord_length[coord])
                # periodic_locs = locations[coord]
                zi = np.array([is_cos*np.cos(k*(zs-left))-is_sin*np.sin(k*(zs-left)) for zs in periodic_locs])

            prod_list.append(np.squeeze(zi))

            
        if(self.dim==3):
            D = np.einsum('ijk,lk,lj,li->li',C,prod_list[2],prod_list[1],prod_list[0],optimize=True)
            D = self.row_comm.allreduce(D)
            D = np.einsum('li->l',D,optimize=True)
            D = self.col_comm.allreduce(D)
        elif(self.dim==2):
            D = np.einsum('ij,lj,li->l',C,prod_list[1],prod_list[0],optimize=True)
            D = comm.allreduce(D)
            
        I = np.real(D)
        return I

    # def initialise_positions(self):
    #     # Initialise using random distributed globally
    #     if(rank==0):
    #         r_vec = np.random.random((self.dim,self.N))
    #     else:
    #         r_vec = np.zeros((self.N,))
    #     r_vec = comm.bcast(r_vec,root=0)

    #     self.positions = np.array([self.coord_boundaries[i][0] + self.coord_length[i]*r_vec[i] for i in range(self.dim)]).T

    def initialise_positions(self, scale=None, loc=None):
                # Initialise using random distributed globally
        if(rank==0):
            r_vec = np.random.random((self.dim,self.N))
        else:
            r_vec = np.zeros((self.N,))
        r_vec = comm.bcast(r_vec,root=0)

        self.positions = np.array([self.coord_boundaries[i][0] + self.coord_length[i]*r_vec[i] for i in range(self.dim)]).T

        if scale != None:
            if(rank==0):
                r_vec = np.array([np.random.normal(loc=loc[i], scale=scale[i], size=(self.N,)) for i in range(self.dim)])

                for i in range(self.dim):
                    r_vec[i,:] = self.coord_boundaries[i][0]+np.mod(r_vec[i,:]-self.coord_boundaries[i][0],self.coord_length[i])
            else:
                r_vec = np.zeros((self.N,))
            r_vec = comm.bcast(r_vec,root=0)
            self.positions = r_vec.T
            self.apply_bcs()

    def get_fluid_vel(self,velocities):

        assert(len(velocities)==self.dim)
        for coord in range(self.dim):
            if(self.dim==3):
                self.fluids_vel[:,coord] = self.interpolate(velocities[coord], (self.positions[:,0], self.positions[:,1],self.positions[:,2]))

            elif(self.dim==2):
                self.fluids_vel[:,coord] = self.interpolate(velocities[coord], (self.positions[:,0], self.positions[:,1]))

    def step(self,dt,velocities):
        self.get_fluid_vel(velocities)

        # Move particles
        self.positions += dt * self.fluids_vel
        self.apply_bcs()

    def apply_bcs(self):
        # Apply BCs on the particle positions
        for coord in range(self.dim):
            if(type(self.bases[coord]).__name__=='RealFourier'):
                pass
                # Periodic boundary conditions
                # self.positions[:,coord] = self.coord_boundaries[coord][0]+np.mod(self.positions[:,coord]-self.coord_boundaries[coord][0],self.coord_length[coord])
            if(type(self.bases[coord]).__name__=='Jacobi'):
                # Non-periodic boundary conditions
                self.positions[:,coord] = np.clip(self.positions[:,coord], self.coord_boundaries[coord][0], self.coord_boundaries[coord][1])
                
    def initialise_stress(self):

        for particle in range(self.N):
            self.J[particle,0,0] = 1./np.sqrt(2)
            self.J[particle,1,1] = 1./np.sqrt(2)

    def get_fluid_stress(self,velocities):
        # Check
        assert(len(velocities)==self.dim)

        for coordi in range(self.dim):
            for coordj in range(self.dim):
                diff_op = self.bases[coordj].Differentiate(velocities[coordi])
                self.S[:,coordi,coordj] = self.interpolate(diff_op, self.positions[:,0], self.positions[:,1],self.positions[:,2])

    def step_stress(self,dt,velocities):
        self.get_fluid_stress(velocities)
        # Move stress
        for particle in range(self.N):
             self.J[particle,:,:] += dt*self.S[particle,:,:]@self.J[particle,:,:]
