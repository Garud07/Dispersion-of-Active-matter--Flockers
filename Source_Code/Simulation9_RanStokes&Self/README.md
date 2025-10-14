# 2d_active_matter
# Simulation9_RanStokes&Self

This repository contains the ninth stage of the *Dispersion of Active Matter in Turbulence* project.  
Here, particles are **randomly initialized** and evolved under **both self-propulsion and Stokesian drag**.  
Unlike previous cases, this setup introduces **random initial velocities and/or orientations**, allowing the study of how stochasticity influences the competition between active and viscous effects.

---

## 📁 Project Structure


---

## ⚙️ Simulation Details

- **Flow Field:** None  
- **Initial Distribution:** Random and uniform  
- **Initial Velocities:** Randomized directions and magnitudes  
- **Dynamics:**  
  - **Self-propulsion:** Active component of velocity with noise  
  - **Stokes drag:** Linear damping (viscous resistance)  
- **Noise:** Random perturbation in orientation/velocity (configurable)  
- **Objective:**  
  To explore emergent order and particle dispersion when random initial conditions interact with competing active and passive dynamics.

---

## 🧠 How to Run

```bash
python main.py
