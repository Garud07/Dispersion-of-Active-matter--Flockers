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

def animate_with_trails(
    locs_file='p_locs.npy',
    times_file='p_times.npy',
    output_file='trails_particles.mp4',
    fps=30,                # Standard video FPS
    duration=20            # seconds
):
    # Load particle data
    locs = np.load(locs_file)  # (N, n_saves, 2)
    times = np.load(times_file)
    N, n_frames, dim = locs.shape

    # Calculate frame_skip for desired duration
    total_required_frames = fps * duration
    frame_skip = max(1, int(np.ceil(n_frames / total_required_frames)))
    locs = locs[:, ::frame_skip, :]
    times = times[::frame_skip]
    n_frames = locs.shape[1]

    print(f"Total particles plotted: {N}")
    print(f"Frames in animation: {n_frames}, frame_skip: {frame_skip}")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(locs[:,:,0].min(), locs[:,:,0].max())
    ax.set_ylim(locs[:,:,1].min(), locs[:,:,1].max())
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Title with more padding (fix cutting)
    title = ax.text(
        0.5, 1.07, "", transform=ax.transAxes, ha="center", va="bottom", fontsize=18
    )

    #trails = [ax.plot([], [], lw=1, color='steelblue', alpha=0.05)[0] for _ in range(N)]
    dots   = [ax.plot([], [], 'o', color='royalblue', markersize=4)[0] for _ in range(N)]

    def init():
        for dot in dots:        #for trail, dot in zip(trails, dots):
            #trail.set_data([], [])
            dot.set_data([], [])
        title.set_text("")
        return dots + [title]         #return trails + dots + [title]

    def update(frame):
        for i in range(N):
            #x = locs[i, :frame+1, 0]
            #y = locs[i, :frame+1, 1]
            x = locs[i, :frame, 0]
            y = locs[i, :frame, 1]
            #trails[i].set_data(x, y)
            dots[i].set_data(x[-1:], y[-1:])
        title.set_text(f"n = {frame+1}/{n_frames}")
        return dots + [title]         #return trails + dots + [title]

    anim = FuncAnimation(
        fig, update, frames=n_frames,
        init_func=init, blit=True, interval=1000/fps
    )
    anim.save(output_file, writer='ffmpeg', fps=fps)
    plt.close(fig)
    print(f"Animation saved as {output_file}")
# Example call:
animate_with_trails(
    locs_file='p_locs.npy',
    times_file='p_times.npy',
    output_file='trails_particles.mp4',
    fps=10,          # standard FPS
    duration=20      # seconds
)
#
# # Load the step‐by‐step times and mean speeds
times = np.load('times_phi.npy')        # shape (10002,)
mean_speeds = np.load('mean_speeds.npy')# shape (10002,)

plt.figure()
plt.plot(times, mean_speeds)
plt.xlabel('Time')
plt.ylabel('Mean Speed ⟨|v|⟩')
plt.title('Mean Particle Velocity vs Time')
plt.tight_layout()
plt.show()

# r_int = 0.3
# times = np.load('times_phi.npy')
# phi   = np.load('order_param_values.npy')
#
# fig, ax = plt.subplots(figsize=(10, 5))
# ax.plot(times, phi, marker='o', markersize=2, linewidth=1.3, label=f"r_int={r_int}")
#
# ax.set_xlabel("Time", fontsize=13)
# # Use plain text, NOT the unicode ϕ
# ax.set_ylabel("Global Order Parameter ($\phi$))", fontsize=13)
# ax.set_title(f"Global Order Parameter vs Time (r_int = {r_int})", fontsize=14)
# ax.grid(True, alpha=0.3)
# ax.legend()
#
# filename = f"phi for r_int = {r_int}.png"
# fig.savefig(filename, dpi=300)
# print(f"Plot saved as {filename}")
#
# plt.show()

