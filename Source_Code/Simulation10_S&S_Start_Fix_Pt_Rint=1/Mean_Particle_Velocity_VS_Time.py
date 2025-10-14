import numpy as np
import matplotlib.pyplot as plt

mean_T10 = np.load("mean_speeds.npy")

# Time step calculation based on known simulation duration
sim_duration = 200  # seconds
n_points = len(mean_T10)
dt = sim_duration / n_points

# Time axes for each simulation
time_T10 = np.arange(len(mean_T10)) * dt

# Analytical solution parameters
v0 = 0.5           # Self-propulsion speed
v_init = 0.2       # Initial Stokes contribution
tau_p = 14         # Decay timescale

# Analytical expression: v(t) = v0 + v_init * exp(-t / tau_p)
t_analytical = np.linspace(0, sim_duration, n_points)
v_analytical = v0 + v_init * np.exp(-t_analytical / tau_p)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(time_T10, mean_T10, label=r'Timestep $T = 10$', lw=2)
plt.plot(t_analytical, v_analytical, 'k--', label=r'Analytical: $v(t) = v_0 + v_i e^{-t/\tau_p}$', lw=2)

plt.xlabel(r'Time $t$ (s)', fontsize=12)
plt.ylabel(r'Mean Speed $\langle |\mathbf{v}| \rangle$', fontsize=12)
plt.title('Mean Particle Speed vs Time with Analytical Comparison', fontsize=13)
plt.legend()
plt.grid(True)
plt.xlim(0, 200)
plt.tight_layout()
plt.show()
