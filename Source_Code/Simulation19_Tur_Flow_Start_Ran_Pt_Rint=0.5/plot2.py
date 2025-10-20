import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helper import create_animation, create_particle_animation
from pyevtk.hl import gridToVTK

fname = './timeseries/timeseries_s1.h5'

with h5py.File(fname, mode='r') as file:
    t = file['tasks']['<omega**2>'].dims[0]['sim_time'][:]
    om = file['tasks']['<omega**2>'][:].squeeze()

plt.plot(t, om)
plt.xlabel('$t$')
plt.ylabel('$<omega**2>$')
plt.show()


fname = './snapshots/snapshots_s1.h5'

with h5py.File(fname, mode='r') as file:
    z = file['tasks']['vorticity'].dims[2][0][:]
    x = file['tasks']['vorticity'].dims[1][0][:]
    omega = file['tasks']['vorticity'][:]

frame = -1
field = omega[frame, :, :]

fig = plt.figure()
plt.pcolormesh(x, z, field.T)
plt.colorbar()
plt.show()

# To create an animation
# create_animation(omega, x=x, z=None, vmin=np.min(omega), vmax=np.max(omega), fps=10)
create_animation(omega, x=x, z=None, vmin=np.min(omega), vmax=np.max(omega), fps=10)

# Particle locations
locs = np.load('p_locs.npy')

# Times (not uniformly spaced)
times = np.load('p_times.npy')

# Plot one particle
#plt.plot(locs[0,:,0], locs[0,:,1], '-k')
#plt.show()

# stride for animation
# stride = 10
# create_particle_animation( locs[:,::stride,:], fps=10, output_file='particles.mp4' )
