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
stop_sim_time = 200
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
omega['g']=0
ex, ey = coords.unit_vector_fields(dist)
u = d3.grad(psi)@ey * ex - d3.grad(psi)@ex * ey
ux = u @ ex
uy = u @ ey

domain = Domain(dist, (xbasis, ybasis))

# Problem
problem = d3.IVP([omega, psi, tau_psi], namespace=locals())
problem.add_equation("dt(omega) - lap(omega)/Re + alpha*omega = -u @ d3.grad(omega) ")
problem.add_equation("lap(psi) + omega + tau_psi = 0")
problem.add_equation("integ(psi) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time
timeseries = solver.evaluator.add_file_handler('timeseries',iter=10)

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

#Initiate particles (N particles)
N = 100
# Parameters for Vicsek/drag model
T     = 10      # alignment interval (timesteps)
v0    = 0.5    # swim speed
r_int = 1      # interaction radius
rho_p = 1      # particle density
a = 0.171    # particle radius
rho_f = 1000.0  # fluid density
nu = 1e-6    # kinematic viscosity
noise = 0.001   # Noise
particle_tracker = particles.particles(
    N, dist, (xbasis, ybasis), T, v0, r_int, rho_p, a, rho_f, nu, noise,
    scale=None,  # Disable Gaussian clustering
    loc=None     # Disable center offset
)

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

    # Initialize storage
    order_param_values = []
    mean_speeds        = []
    times_phi          = []
    locs               = []
    times              = []

    savet  = 0.0       # Next save‐time for sparse output
    savedt = 0.01      # Interval for sparse position saves

    while solver.proceed:

        timestep = CFL.compute_timestep()
        solver.step(timestep)

        #f['c'] += -lam * f['c'] * timestep + (2 * lam) ** (1 / 2) * forcing(domain, A) * np.sqrt(timestep)
        particle_tracker.step(timestep, (ux, uy))

        # 1) Stash old particle positions
        prev_pos = particle_tracker.positions.copy()

        # 2) Advance fluid + particles
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        particle_tracker.step(timestep, (ux, uy))

        # 3) Compute & store mean speed
        disp   = particle_tracker.positions - prev_pos
        speeds = np.linalg.norm(disp, axis=1) / timestep
        mean_speeds.append(speeds.mean())
        times_phi.append(solver.sim_time)

        # 4) Compute & store order parameter
        phi = particle_tracker.global_order_parameter()
        order_param_values.append(phi)

        # 5) Sparse save of positions & times
        if solver.sim_time >= savet:
            locs.append(particle_tracker.positions.copy())
            times.append(solver.sim_time)
            savet += savedt

        # 6) Log flow stats
        omega2   = flow.max('<omega**2>')
        domega2  = flow.max('<|d omega|**2>')
        logger.info(
            f"n={solver.iteration}, t={solver.sim_time:.4f}, dt={timestep:.2e}, "
            f"<omega**2>={omega2:.4e}, <|d omega|**2>={domega2:.4e}, "
            f"order_param={phi:.6f}"
        )

    # After loop: print average order parameter
    avg_phi = sum(order_param_values) / len(order_param_values)
    print(f"Average order parameter over simulation: {avg_phi:.6f}")

except Exception:
    logger.error('Exception raised, triggering end of main loop.')
    raise

finally:
    solver.log_stats()

    # Re-shape & save data (only on rank 0)
    locs = np.array(locs)
    locs = np.transpose(locs, axes=(1, 0, 2))

    if rank == 0:
        np.save('p_locs.npy',      locs)
        np.save('p_times.npy',     times)
        np.save('times_phi.npy',   np.array(times_phi))
        np.save('mean_speeds.npy', np.array(mean_speeds))
        np.save('order_param_values.npy', np.array(order_param_values))







