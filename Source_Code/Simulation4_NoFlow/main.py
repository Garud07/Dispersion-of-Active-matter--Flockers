import copy
import numpy as np
import dedalus.public as d3
import logging
import time
import particles
import trace
import sys
from docopt import docopt

# from dedalus.extras.plot_tools import plot_bot_2d
from mpi4py import MPI
from dedalus.tools import post
from dedalus.core.domain import Domain

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger(__name__)

# Parameters
Lx, Ly = 2*np.pi, 2*np.pi
Nx, Ny = 64, 64

# Inverse timescale

Re = 1e3
alpha = .01
lam = 1

dealias = 3/2
stop_sim_time = 100
timestepper = d3.RK222
max_timestep = 2e-2
dtype = np.float64

Nx_dealias = int(dealias * Nx)
Ny_dealias = int(dealias * Ny)

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(-Lx/2, Lx/2), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

# Fields
omega = dist.Field(name='omega', bases=(xbasis,ybasis))
#f  = dist.Field(name='f', bases=(xbasis,ybasis))
psi   = dist.Field(name='psi', bases=(xbasis,ybasis))
tau_psi = dist.Field(name='tau_psi')
t = dist.Field()

# Substitutions
x, y = dist.local_grids(xbasis, ybasis)
omega['g'] = 0
ex, ey = coords.unit_vector_fields(dist)
u = d3.grad(psi)@ey * ex - d3.grad(psi)@ex * ey
ux = u @ ex
uy = u @ ey

domain = Domain(dist, (xbasis, ybasis))

# Problem
problem = d3.IVP([omega, psi, tau_psi], namespace=locals())
problem.add_equation("dt(omega) - lap(omega)/Re = -u @ d3.grad(omega)")
problem.add_equation("lap(psi) + omega + tau_psi = 0")
problem.add_equation("integ(psi) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#psi.fill_random('g', seed=42, distribution='normal', scale=1e-2, ) # Random noise
#psi.low_pass_filter(scales=0.1)
#omega.change_scales(dealias)
#omega['g'] = -d3.lap(psi).evaluate()['g']

#def initialize_forcing(domain, k_range = [3, 4]):
    # Use Dedalus tools to access local wavenumber arrays

 #   kx, ky = np.meshgrid(domain.bases[0].wavenumbers, domain.bases[1].wavenumbers, indexing='ij')
    # Forcing amplitude function (example: uniform over band)

  #  ksq = kx**2 + ky**2
   # A = np.logical_and(ksq >= k_range[0]**2, ksq <= k_range[1]**2).astype(float)

    # Normalize (avoid zero-division)
    #A /= np.sum(A)

    # Store for later use
    #return A

#def forcing(domain, A):
    # Get local shapes
 #   cslices = domain.dist.coeff_layout.slices(domain,scales=1)
  #  crand = np.random.RandomState(seed=42)
   # cshape = domain.dist.coeff_layout.global_shape(domain, scales=1)
   # return A[cslices] * np.random.normal(size=cshape)[cslices]


# Initialise forcing
#f['c'] = 0.0

#A = initialize_forcing(domain)

# Time Series
#timeseries = solver.evaluator.add_file_handler('timeseries',iter=10)

# Spectra
spectra = solver.evaluator.add_file_handler('spectra', sim_dt=1)
spectra.add_task(u, layout='c', name='Eu(k)')

# 2d Fields
analysis_tasks = []
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.5, max_writes=200)

snapshots.add_task(omega, name='vorticity', scales=dealias)
analysis_tasks.append(snapshots)

# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.1, threshold=0.05,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=100)
flow.add_property(d3.Integrate(omega**2), name='<omega**2>')
flow.add_property(d3.Integrate(d3.grad(omega)@d3.grad(omega)), name='<|d omega|**2>')
#flow.add_property(d3.Integrate(omega * f), name='<omega*f>')

#Initiate particles (N particles)
N = 1000
# Parameters for Vicsek/drag model
T     = 2       # alignment interval (timesteps)
v0    = 0.1      # swim speed
r_int = 1      # interaction radius
rho_p = 2000.0   # particle density
a     = 1e-3     # particle radius
rho_f = 1000.0   # fluid density
nu    = 1e-6     # kinematic viscosity
noise = 0.001      # Noise
particle_tracker = particles.particles(N,dist,(xbasis, ybasis),T, v0, r_int, rho_p, a, rho_f, nu,noise, scale=(0.05,0.05), loc=(0,0))

locs = []
pos = copy.copy(particle_tracker.positions)
locs.append(pos)
savet = 0
savedt = 0.01
times = [0.]
savet += savedt

# Main loop
try:
    logger.info('Starting main loop')
    start_time = time.time()
    order_param_values = []
    locs = []
    times = []

    savet = 0    # Initial save time
    savedt = 0.01 # Jitna interval pe save karna hai (define this before loop)


    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        particle_tracker.step(timestep, (ux, uy))

        #f['c'] += -lam * f['c'] * timestep + (2 * lam) ** (1 / 2) * forcing(domain, A) * np.sqrt(timestep)

        # Calculate and store order parameter
        phi = particle_tracker.global_order_parameter()
        order_param_values.append(phi)

        # Save positions/times if needed
        if solver.sim_time >= savet:
            pos = copy.copy(particle_tracker.positions)
            locs.append(pos)
            times.append(solver.sim_time)
            savet += savedt

        # Compute omega quantities every step
        omega2 = flow.max('<omega**2>')
        domega2 = flow.max('<|d omega|**2>')
        #omegaf = flow.max('<omega*f>')

        # Print/log everything every iteration
        logger.info(
            f'n={solver.iteration}, t={solver.sim_time:.4f}, dt={timestep:.2e}, '
            f'<omega**2>={omega2:.4e}, <|d omega|**2>={domega2:.4e}, '
            #f'<omega*f>={omegaf:.4e}, order_param={phi:.6f}'
        )


    # After loop: print average order parameter (if needed)
    avg_phi = sum(order_param_values) / len(order_param_values)
    print(f"Average order parameter over simulation: {avg_phi:.6f}")

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

    locs = np.array(locs)
    locs = np.transpose(locs, axes=(1, 0, 2))

    if rank == 0:
        np.save('p_locs', locs)
        np.save('p_times', times)






