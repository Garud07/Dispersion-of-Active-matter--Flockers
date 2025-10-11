import h5py
import numpy as np
import matplotlib.pyplot as plt

filename = "/home/shara/miniforge3/Project/2d_active_matter-main/Source_Code/Simulation1/snapshots/snapshots_s1.h5"

with h5py.File(filename, "r") as file:
    print("Top-level keys:", list(file.keys()))
    print("Keys in 'tasks':", list(file['tasks'].keys()))

    dataset = file['tasks']['vorticity']  # ya jo bhi naam aaye
    print("Shape:", dataset.shape)

    # Agar shape (11, 128, 128) hai:
    data = dataset[-1, :, :]  # ✅

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(data, origin='lower', aspect='auto', cmap='viridis')
    plt.colorbar(label='vorticity')
    plt.title('Simulation Output')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.tight_layout()
    plt.show()
