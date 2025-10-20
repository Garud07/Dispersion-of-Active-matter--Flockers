"""
Run interpolation tests

Usage:
    parallelTest.py [--mesh=<mesh> --N=<N>]

Options:
    --mesh=<mesh>              Processor mesh for 3-D runs
    --N=<N>                    Number of particles [default: 128]
"""

import particles
import dedalus.public as de
import numpy as np
from dedalus.core import field
from mpi4py import MPI
import time
from docopt import docopt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
dtype = np.float64
dealias = 3/2
args = docopt(__doc__)

N    = int(args['--N'])
mesh = args['--mesh']

if(rank==0):
    print("All tests running with {} particles".format(N))
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
    if(rank==0):
        print("3D tests with processor mesh={}".format(mesh))
else:
    log2 = np.log2(size)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    if(rank==0):
        print("3D tests with processor mesh={}".format(mesh))

Lx, Ly, Lz = (2*np.pi, 2*np.pi, 2*np.pi)
offx, offy, offz = (0,0,0)
nx, ny, nz = (64,64,64)

coords = de.CartesianCoordinates('x', 'y', 'z')
dist = de.Distributor(coords, dtype=dtype, mesh=mesh)
x_basis = de.RealFourier(coords['x'], size=nx, bounds=(offx, offx+Lx), dealias=dealias)
y_basis = de.RealFourier(coords['y'], size=ny, bounds=(offy, offy+Ly), dealias=dealias)
z_basis = de.Chebyshev(coords['z'],   size=nz, bounds=(offz, offz+Lz), dealias=dealias)

x,y,z = dist.local_grids(x_basis, y_basis, z_basis)
f = dist.Field(name='f', bases=(x_basis,y_basis,z_basis))

p = particles.particles(N,dist,(x_basis, y_basis, z_basis))


f['g'] = np.random.rand(*f['g'].shape)

f['g'] = z**2

# for pos in p.positions:
#     print(rank, pos, p.dist.mesh)


# # Interpolate f at the particle positions using dedalus

# nTime = time.time()
# newInterp = p.interpolate(f,(p.positions[:,0],p.positions[:,1],p.positions[:,2]))
# nTime = time.time() - nTime

for pos in p.positions:
    print('rank {:n}: {:s}'.format(rank, pos.__repr__()))


t_custom = time.time()
f_custom = p.interpolate(f,(p.positions[:,0],p.positions[:,1],p.positions[:,2]))
t_custom = time.time() - t_custom

# if (rank==0):
#     t_custom = time.time()
#     f_custom = p.interpolate(f,(p.positions[:,0],p.positions[:,1],p.positions[:,2]))
#     t_custom = time.time() - t_custom

t_default = time.time()
f_default = []

for pos in p.positions:
    f_default.append(f(x=pos[0], y=pos[1], z=pos[2]).evaluate()['g'])
    
t_default = time.time() - t_default

if (rank==0):
    f_default = np.squeeze(f_default)
    print('L1 error: {:7.4g}\nSpeed up: {:5.4g}'.format(np.mean(np.abs(f_default-f_custom)),t_default/t_custom))

