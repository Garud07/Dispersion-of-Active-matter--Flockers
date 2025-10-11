import h5py
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from helper import create_animation, create_particle_animation
from pyevtk.hl import gridToVTK
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
# Particle locations
locs = np.load('p_locs.npy')
# Times (not uniformly spaced)
times = np.load('p_times.npy')
#Plot one particle
plt.plot(locs[1,:,0], locs[1,:,1], '-k')
plt.show()


#plt.xlabel("x")
#plt.ylabel("y")
#plt.title("Trajectories of 10 Particles with Direction Arrows")
#plt.grid(True)
#plt.show()



# stride for animation
# stride = 10
# create_particle_animation( locs[:,::stride,:], fps=10, output_file='particles.mp4' )




#locs = np.load('p_locs.npy')
#stride = 20  # arrow every 20th step
#plt.figure(figsize=(10, 7))
#for i in range(10):  # only 10 particles
 #   x = locs[i, ::stride, 0]
  #  y = locs[i, ::stride, 1]
   # u = np.diff(x, prepend=x[0])
    #v = np.diff(y, prepend=y[0])
    #plt.quiver(x, y, u, v, angles='xy', scale_units='xy', scale=1, width=0.002)



