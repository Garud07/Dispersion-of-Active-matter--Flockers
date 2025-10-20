import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')

from matplotlib.animation import FuncAnimation

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def create_animation(data, x=None, z=None, output_file='animation.mp4', fps=5, colormap='viridis', vmin=None, vmax=None):
    """
    Creates an animation from a 3D NumPy array and saves it to a file, with non-uniform coordinates and an adjustable color scale.
    
    Parameters:
        data (np.ndarray): A 3D NumPy array of shape (M, Nx, Nz).
        x (np.ndarray): 1D array of shape (Nx,) representing the x-coordinates (not necessarily uniform).
        z (np.ndarray): 1D array of shape (Nz,) representing the z-coordinates (not necessarily uniform).
        output_file (str): The name of the output animation file (e.g., 'animation.mp4').
        fps (int): Frames per second for the output video.
        colormap (str): Colormap for the visualization.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D NumPy array of shape (M, Nx, Nz).")
    
    M, Nx, Nz = data.shape

    # Set default coordinates if not provided
    if x is None:
        x = np.linspace(0, Nx, Nx)
    if z is None:
        z = np.linspace(0, Nz, Nz)

    if len(x) != Nx or len(z) != Nz:
        raise ValueError("x and z must have lengths matching the data dimensions Nx and Nz.")

    # Create a grid for the coordinates (cell boundaries for pcolormesh)
    x_edges = np.linspace(x[0], x[-1], Nx + 1) if np.ptp(np.diff(x)) > 1e-8 else np.append(x, x[-1] + np.diff(x)[0])
    z_edges = np.linspace(z[0], z[-1], Nz + 1) if np.ptp(np.diff(z)) > 1e-8 else np.append(z, z[-1] + np.diff(z)[0])

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Initialize the plot with pcolormesh
    mesh = ax.pcolormesh(z_edges, x_edges, data[0], cmap=colormap, shading='auto', vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label('')
    ax.set_title(f"n = 1/{M}")
    ax.set_xlabel("z")
    ax.set_ylabel("x")

    # Animation update function
    def update(frame):
        mesh.set_array(data[frame].ravel())  # Update the data
        ax.set_title(f"Frame {frame+1}/{M}")
        return [mesh]

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=M, interval=1000//fps, blit=True
    )

    # Save the animation
    ani.save(output_file, fps=fps)
    print(f"Animation saved to {output_file}")


    
def create_particle_animation(data, x=None, z=None, output_file='animation.mp4', fps=5, dpi=200):
    """
    Creates an animation from a 3D NumPy array and saves it to a file, with non-uniform coordinates and an adjustable color scale.
    
    Parameters:
        data (np.ndarray): A 3D NumPy array of shape (N, M, D).
        output_file (str): The name of the output animation file (e.g., 'animation.mp4').
        fps (int): Frames per second for the output video.
        colormap (str): Colormap for the visualization.
    """
    if data.ndim != 3:
        raise ValueError("Input data must be a 3D NumPy array of shape (M, Nx, Nz).")

    N, M, D = data.shape

    # Axis limits
    xlim = [np.min(data[:, :, 0]), np.max(data[:, :, 0])]
    ylim = [np.min(data[:, :, 1]), np.max(data[:, :, 1])]

    # Initialize figure and axis
    fig, ax = plt.subplots()
    lines = [ax.plot([], [], '-',color='lightgrey',lw=0.4)[0] for _ in range(N)]
    points = [ax.plot([], [], 'oC0', ms=2)[0] for _ in range(N)]
    title = ax.set_title("")

    # Initialize function
    def init():
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        return lines + points + [title]

    # Update function
    def update(frame):
        frame = min(frame, M - 1)  # Prevent out-of-bounds error
        for i in range(N):
            if frame == 0:
                lines[i].set_data([data[i, 0, 0]], [data[i, 0, 1]])  # Single point at start
            else:
                lines[i].set_data(data[i, :frame+1, 0].ravel(), data[i, :frame+1, 1].ravel())
                points[i].set_data([data[i, frame, 0]], [data[i, frame, 1]])  # Single point as list
                title.set_text(f'n = {frame+1}/{M}')
        return lines + points + [title]

    # Ensure fps is defined
    fps = 30  

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=M, init_func=init, interval=1000//fps, blit=True
    )

    # Save the animation
    ani.save(output_file, fps=fps, dpi=dpi)
    print(f"Animation saved to {output_file}")



    
