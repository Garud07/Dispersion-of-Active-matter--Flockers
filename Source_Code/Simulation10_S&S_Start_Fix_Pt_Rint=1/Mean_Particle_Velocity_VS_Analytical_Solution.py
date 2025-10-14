import numpy as np
import matplotlib.pyplot as plt

# Files ka path (agar same folder hai toh direct naam likho)
phi_T1  = np.load('order_param_values(T=1,200sec).npy')
phi_T3  = np.load('order_param_values(T=3,200sec).npy')
phi_T5  = np.load('order_param_values(T=5,200sec).npy')
phi_T15 = np.load('order_param_values(T=15,200sec).npy')

# Har ek ke liye time array banao
t1   = np.linspace(0, 200, len(phi_T1))
t3   = np.linspace(0, 200, len(phi_T3))
t5   = np.linspace(0, 200, len(phi_T5))
t15  = np.linspace(0, 200, len(phi_T15))

plt.figure(figsize=(10,5))
plt.plot(t1, phi_T1, label=r"$T=1$")
plt.plot(t3, phi_T3, label=r"$T=3$")
plt.plot(t5, phi_T5, label=r"$T=5$")
plt.plot(t15, phi_T15, label=r"$T=15$")

plt.xlabel(r'Time (s)', fontsize=13)
plt.ylabel(r'Global Order Parameter ($\phi$)', fontsize=13)
plt.title(r'Global Order Parameter vs Time ($r_{\mathrm{int}} = 0.3$)', fontsize=15)
plt.legend(title=r'Alignment Interval ($T$)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("phi_vs_time_rint_0_3_latex_multiT_4curves.png", dpi=300)
plt.show()

